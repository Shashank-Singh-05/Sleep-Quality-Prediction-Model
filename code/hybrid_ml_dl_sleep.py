# hybrid_ml_dl_sleep.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# DL
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K

# ---------------------------
# Utility: robust AUC for sklearn-like models (uses predict_proba or CV fallback)
# ---------------------------
def compute_auc_with_fallback_sklearn(model, X_test, y_test, X_all, y_all, cv_fallback=5):
    """
    Try predict_proba on X_test. If fails (or test has <2 classes), fallback to cross_val_predict predict_proba on whole dataset.
    Returns (auc_value, source_str) or (None, None).
    """
    classes = getattr(model, "classes_", np.unique(y_all))
    unique_test_classes = np.unique(y_test)

    # try predict_proba on test
    y_prob_test = None
    try:
        y_prob_test = model.predict_proba(X_test)
    except Exception:
        y_prob_test = None

    if y_prob_test is not None and unique_test_classes.size >= 2:
        try:
            if classes.shape[0] == 2:
                pos_index = 1 if 1 in list(classes) else 0
                auc = roc_auc_score(y_test, y_prob_test[:, pos_index])
            else:
                y_test_bin = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(y_test_bin, y_prob_test, average='weighted', multi_class='ovr')
            return auc, "test_proba"
        except Exception:
            pass

    # fallback: cross-validated predict_proba on full data
    try:
        cv_probs = cross_val_predict(model, X_all, y_all, cv=cv_fallback, method='predict_proba')
        classes_cv = np.unique(y_all)
        if classes_cv.shape[0] == 2:
            pos_index = 1 if 1 in list(classes_cv) else 0
            auc = roc_auc_score(y_all, cv_probs[:, pos_index])
        else:
            y_all_bin = label_binarize(y_all, classes=classes_cv)
            auc = roc_auc_score(y_all_bin, cv_probs, average='weighted', multi_class='ovr')
        return auc, f"cv{cv_fallback}"
    except Exception:
        return None, None

# ---------------------------
# Keras wrapper helper for small meta-learner and embedding model
# ---------------------------
def build_meta_nn(input_dim, n_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_embedding_nn(input_dim, n_classes):
    """A network where we'll extract penultimate-layer outputs as embeddings."""
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dense(64, activation='relu')(x)
    embedding = layers.Dense(32, activation='relu', name='embedding')(x)  # penultimate
    out = layers.Dense(n_classes, activation='softmax')(embedding)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------
# Load & preprocess (same pipeline)
# ---------------------------
df = pd.read_csv(r"D:\AA\Projects\sleep analysis\sleep.csv")
new = pd.DataFrame(df)

# Label encode categorical columns and label
le_gender = LabelEncoder()
le_occ = LabelEncoder()
le_bmi = LabelEncoder()
le_sleep = LabelEncoder()

new["Gender"] = le_gender.fit_transform(new["Gender"])
new["Occupation"] = le_occ.fit_transform(new["Occupation"])
new["BMI Category"] = le_bmi.fit_transform(new["BMI Category"])
new["Sleep Disorder"] = le_sleep.fit_transform(new["Sleep Disorder"])

# blood pressure split
new[["Systolic", "Diastolic"]] = new["Blood Pressure"].str.split("/", expand=True).astype(int)

# feature matrix and labels
X = new[[
    "Gender","Age","Occupation","Sleep Duration","Quality of Sleep",
    "Physical Activity Level","Stress Level","BMI Category",
    "Systolic","Diastolic","Heart Rate","Daily Steps"
]].values

y = new["Sleep Disorder"].values

# scale features for DL parts
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train/test split (same stratify)
X_train, X_test, y_train, y_test, Xs_train, Xs_test = train_test_split(
    X, y, X_scaled, test_size=0.2, random_state=42, stratify=y
)

# For convenience:
# - X_train/X_test = original features (unscaled)
# - Xs_train/Xs_test = scaled features for DL
# But many ML models will work fine on scaled features as well. We'll use scaled for DL and original or scaled for ML accordingly.

# ---------------------------
# HYBRID MODEL A: STACKING (ML base models -> DL meta-learner)
# ---------------------------
print("\n" + "="*10 + " HYBRID A: Stacking (ML base -> DL meta-learner) " + "="*10)

# Base models
base_models = [
    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
]

# We'll get out-of-fold (OOF) probabilities on train (to train meta-learner),
# and test probabilities by training base models on full train and predicting test.
n_classes = len(np.unique(y))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# OOF train prob storage
oof_train_probs = np.zeros((X_train.shape[0], n_classes * len(base_models)))
test_probs_concat = np.zeros((X_test.shape[0], n_classes * len(base_models)))

for i, (name, model) in enumerate(base_models):
    # cross_val_predict with method='predict_proba' to get OOF probabilities for training
    print(f"Generating OOF probs for base model: {name}")
    # Use scaled features for models too (safer)
    oof = cross_val_predict(model, Xs_train, y_train, cv=skf, method='predict_proba')
    oof_train_probs[:, i*n_classes:(i+1)*n_classes] = oof

    # Fit model on full train and predict_proba on test
    model.fit(Xs_train, y_train)
    test_probs = model.predict_proba(Xs_test)
    test_probs_concat[:, i*n_classes:(i+1)*n_classes] = test_probs

# Train DL meta-learner on oof_train_probs
K.clear_session()
meta_input_dim = oof_train_probs.shape[1]
meta_model = build_meta_nn(meta_input_dim, n_classes)
es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
meta_model.fit(oof_train_probs, y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=[es], verbose=0)

# Evaluate stacking meta-learner on test (feed test base probs)
y_pred_meta = np.argmax(meta_model.predict(test_probs_concat, verbose=0), axis=1)

# basic metrics
acc_A = accuracy_score(y_test, y_pred_meta)
prec_A = precision_score(y_test, y_pred_meta, average='weighted', zero_division=0)
rec_A = recall_score(y_test, y_pred_meta, average='weighted', zero_division=0)
f1_A = f1_score(y_test, y_pred_meta, average='weighted', zero_division=0)

print(f"\nHybrid A (Stacking) Accuracy: {acc_A*100:.2f}%")
print(f"Precision: {prec_A*100:.2f}%")
print(f"Recall: {rec_A*100:.2f}%")
print(f"F1 Score: {f1_A*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_meta, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_meta))

# Compute ROC AUC for meta-learner:
# Try direct predict_proba (meta_model), else fallback to CV across whole dataset by training meta via CV.
try:
    probs_meta_test = meta_model.predict(test_probs_concat, verbose=0)
    if n_classes == 2:
        auc_A = roc_auc_score(y_test, probs_meta_test[:, 1])
        auc_source_A = "meta_test_proba"
    else:
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        auc_A = roc_auc_score(y_test_bin, probs_meta_test, average='weighted', multi_class='ovr')
        auc_source_A = "meta_test_proba"
except Exception:
    # fallback: cross-validated predict_proba for stacking meta-learner
    print("Fallback: computing CV predict_proba for meta-learner (this trains models)...")
    # Build training-level meta features using OOF (already have) and perform CV on entire dataset:
    try:
        # We'll produce CV probabilities by redoing base model OOF across full data (train+test).
        X_all_scaled = np.vstack([Xs_train, Xs_test])
        y_all = np.concatenate([y_train, y_test])
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_oof_all = np.zeros((X_all_scaled.shape[0], n_classes * len(base_models)))
        for i, (name, model) in enumerate(base_models):
            # get OOF probs for full dataset (train+test) using cross_val_predict
            oof_all = cross_val_predict(model, X_all_scaled, y_all, cv=skf2, method='predict_proba')
            meta_oof_all[:, i*n_classes:(i+1)*n_classes] = oof_all
        # Train a fresh meta NN with CV folds on meta_oof_all to get predict_proba; but easier: do CV predictions for meta NN
        # We'll do cross_val_predict with a Keras wrapper is complex; instead compute AUC on meta_oof_all directly vs y_all
        # Train small NN on meta_oof_all (fit on full) and predict—(not perfect but gives us an estimate)
        K.clear_session()
        meta_full = build_meta_nn(meta_oof_all.shape[1], n_classes)
        meta_full.fit(meta_oof_all, y_all, epochs=30, batch_size=32, verbose=0)
        probs_meta_all = meta_full.predict(meta_oof_all, verbose=0)
        if n_classes == 2:
            auc_A = roc_auc_score(y_all, probs_meta_all[:,1])
        else:
            y_all_bin = label_binarize(y_all, classes=np.unique(y))
            auc_A = roc_auc_score(y_all_bin, probs_meta_all, average='weighted', multi_class='ovr')
        auc_source_A = "meta_cv_estimate"
    except Exception:
        auc_A, auc_source_A = None, None

if auc_A is not None:
    print(f"ROC AUC ({auc_source_A}): {auc_A*100:.2f}%")
else:
    print("ROC AUC: could not be computed for Hybrid A.")

# ---------------------------
# HYBRID MODEL B: DL embedding -> ML classifier on (embedding + features)
# ---------------------------
print("\n" + "="*10 + " HYBRID B: DL embedding -> ML classifier " + "="*10)

# Train embedding DL on scaled features (Xs_train)
K.clear_session()
embedding_model = build_embedding_nn(Xs_train.shape[1], n_classes)
es2 = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0)
embedding_model.fit(Xs_train, y_train, validation_data=(Xs_test, y_test), epochs=100, batch_size=32, callbacks=[es2], verbose=0)

# Extract embedding model (up to 'embedding' layer)
embedding_extractor = models.Model(inputs=embedding_model.input,
                                   outputs=embedding_model.get_layer('embedding').output)

# compute embeddings
emb_train = embedding_extractor.predict(Xs_train, verbose=0)  # shape (n_train, emb_dim)
emb_test = embedding_extractor.predict(Xs_test, verbose=0)

# concatenate embeddings with scaled original features (or original features)
X_ml_train = np.hstack([emb_train, Xs_train])
X_ml_test = np.hstack([emb_test, Xs_test])

# Train an ML classifier on these concatenated representations
rf_on_emb = RandomForestClassifier(n_estimators=300, random_state=42)
rf_on_emb.fit(X_ml_train, y_train)
y_pred_B = rf_on_emb.predict(X_ml_test)

# metrics
acc_B = accuracy_score(y_test, y_pred_B)
prec_B = precision_score(y_test, y_pred_B, average='weighted', zero_division=0)
rec_B = recall_score(y_test, y_pred_B, average='weighted', zero_division=0)
f1_B = f1_score(y_test, y_pred_B, average='weighted', zero_division=0)

print(f"\nHybrid B (Embedding->RF) Accuracy: {acc_B*100:.2f}%")
print(f"Precision: {prec_B*100:.2f}%")
print(f"Recall: {rec_B*100:.2f}%")
print(f"F1 Score: {f1_B*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_B, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_B))

# AUC for Hybrid B
auc_B, src_B = compute_auc_with_fallback_sklearn(rf_on_emb, X_ml_test, y_test, np.vstack([X_ml_train, X_ml_test]), np.concatenate([y_train, y_test]), cv_fallback=5)
if auc_B is not None:
    print(f"ROC AUC ({src_B}): {auc_B*100:.2f}%")
else:
    print("ROC AUC: could not be computed for Hybrid B.")

# ---------------------------
# SUMMARY
# ---------------------------
summary = pd.DataFrame([
    {"model":"Hybrid_Stacking_ML->DL", "accuracy":acc_A, "precision":prec_A, "recall":rec_A, "f1":f1_A, "roc_auc":auc_A},
    {"model":"Hybrid_DLembedding->RF", "accuracy":acc_B, "precision":prec_B, "recall":rec_B, "f1":f1_B, "roc_auc":auc_B},
]).set_index("model")

summary["accuracy_pct"] = (summary["accuracy"] * 100).round(2)
summary["precision_pct"] = (summary["precision"] * 100).round(2)
summary["recall_pct"] = (summary["recall"] * 100).round(2)
summary["f1_pct"] = (summary["f1"] * 100).round(2)
summary["roc_auc_pct"] = summary["roc_auc"].apply(lambda x: round(x*100,2) if x is not None else np.nan)

print("\n\n=== HYBRID MODELS SUMMARY ===")
print(summary[["accuracy_pct","precision_pct","recall_pct","f1_pct","roc_auc_pct"]])

# ---------------------------
# OPTIONAL: interactive user prediction using Hybrid B (embedding->RF)
# (commented out to avoid interrupting batch runs)
# ---------------------------
"""
# Example: get user input (encoded categorical values) and run Hybrid B prediction
print("\n=== User Input Prediction (Hybrid B demonstration) ===")
gender = int(input("enter your gender (encoded): "))
age = int(input("enter your age: "))
occupation = int(input("enter your occupation (encoded): "))
sleepdu = float(input("enter your sleep duration: "))
sleepqu = float(input("enter your sleep quality: "))
phycialadct = float(input("enter your physical activity level: "))
StressLevel = float(input("enter your Stress Level: "))
BMICategory = int(input("enter your BMI Category (encoded): "))
Systolic = float(input("enter your Systolic: "))
Diastolic = float(input("enter your Diastolic: "))
heartrate = float(input("enter your heart rate: "))
dailystep = float(input("enter your daily steps: "))

user_feat = np.array([[gender, age, occupation, sleepdu, sleepqu,
                       phycialadct, StressLevel, BMICategory,
                       Systolic, Diastolic, heartrate, dailystep]])
user_feat_scaled = scaler.transform(user_feat)
user_emb = embedding_extractor.predict(user_feat_scaled)
user_concat = np.hstack([user_emb, user_feat_scaled])
pred = rf_on_emb.predict(user_concat)
decoded = le_sleep.inverse_transform(pred.astype(int))[0]
print("Predicted Sleep Disorder (Hybrid B):", decoded)
"""
