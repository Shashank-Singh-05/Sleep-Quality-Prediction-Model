import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from tensorflow.keras.utils import to_categorical


class KerasClassifierWrapper:
    """
    Minimal sklearn-like wrapper around a Keras model builder function.
    The builder_fn must accept (input_dim, n_classes) and return a compiled tf.keras.Model.
    """
    def __init__(self, builder_fn, input_dim=None, n_classes=None, epochs=50, batch_size=32, verbose=0, **fit_kwargs):
        self.builder_fn = builder_fn
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.fit_kwargs = fit_kwargs
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y, validation_data=None):
        # set classes_
        self.classes_ = np.unique(y)
        self.input_dim = X.shape[1] if self.input_dim is None else self.input_dim
        self.n_classes = len(self.classes_) if self.n_classes is None else self.n_classes

        # clear session to avoid clutter
        K.clear_session()
        self.model_ = self.builder_fn(self.input_dim, self.n_classes)

        # callbacks: early stopping + reduce lr
        cb = [
            callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, verbose=0)
        ]
        # Fit
        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_data=validation_data,
            callbacks=cb,
            **self.fit_kwargs
        )
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        probs = self.model_.predict(X, verbose=0)
        # If binary with single output, convert to two-column probs
        if probs.ndim == 1 or (probs.shape[1] == 1 and self.n_classes == 2):
            # ensure shape (n_samples, 2)
            probs = np.vstack([1-probs.ravel(), probs.ravel()]).T
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# ---------------------------
# DL model builders
# ---------------------------
def build_dense_small(input_dim, n_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_dense_deep(input_dim, n_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_wide_deep(input_dim, n_classes):
    inp = layers.Input(shape=(input_dim,))
    # wide branch (shallow)
    wide = layers.Dense(64, activation='relu')(inp)
    # deep branch (deeper)
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    merged = layers.concatenate([wide, x])
    x = layers.Dense(64, activation='relu')(merged)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_dense_bn_dropout(input_dim, n_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.25)(x)

    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------
# Robust ROC AUC computation with manual CV fallback
# ---------------------------
def compute_auc_with_fallback_dl(wrapper, X_test, y_test, X_all, y_all, cv_folds=5, epochs_cv=40, batch_size_cv=32):
    """
    wrapper: a KerasClassifierWrapper instance (with builder_fn available)
    Tries predict_proba on X_test. If not possible (or test has <2 classes), performs
    manual StratifiedKFold cross-validated predict_proba using the wrapper's builder_fn.
    """
    # try test probs
    try:
        probs_test = wrapper.predict_proba(X_test)
    except Exception as e:
        probs_test = None

    unique_test_classes = np.unique(y_test)
    classes = np.unique(y_all)

    if (probs_test is not None) and (unique_test_classes.size >= 2):
        try:
            if classes.shape[0] == 2:
                pos_index = 1 if 1 in classes else 0
                return roc_auc_score(y_test, probs_test[:, pos_index]), "test_proba"
            else:
                y_test_bin = label_binarize(y_test, classes=classes)
                return roc_auc_score(y_test_bin, probs_test, average='weighted', multi_class='ovr'), "test_proba"
        except Exception:
            pass

    # Fallback: manual stratified CV to get probabilities for all samples
    print("Falling back to manual stratified CV predict_proba for AUC (this trains cv models)...")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    probs = np.zeros((X_all.shape[0], classes.shape[0]))
    for train_idx, val_idx in skf.split(X_all, y_all):
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        # build & train a fresh model using the same builder function as wrapper
        K.clear_session()
        model = wrapper.builder_fn(X_all.shape[1], len(classes))  # compile already done in builder
        es = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=0)
        model.fit(X_tr, y_tr, epochs=epochs_cv, batch_size=batch_size_cv, verbose=0,
                  validation_data=(X_val, y_val), callbacks=[es])
        p_val = model.predict(X_val, verbose=0)
        # ensure proper shape
        if p_val.ndim == 1 or (p_val.shape[1] == 1 and len(classes) == 2):
            p_val = np.vstack([1-p_val.ravel(), p_val.ravel()]).T
        probs[val_idx] = p_val

    # compute AUC on whole dataset probs
    try:
        if classes.shape[0] == 2:
            pos_index = 1 if 1 in classes else 0
            auc_value = roc_auc_score(y_all, probs[:, pos_index])
        else:
            y_all_bin = label_binarize(y_all, classes=classes)
            auc_value = roc_auc_score(y_all_bin, probs, average='weighted', multi_class='ovr')
        return auc_value, f"manual_cv{cv_folds}"
    except Exception as e:
        print("Manual CV AUC failed:", e)
        return None, None

# ---------------------------
# Load & preprocess (mirror your pipeline)
# ---------------------------
df = pd.read_csv(r"D:\AA\Projects\sleep analysis\sleep.csv")
new = pd.DataFrame(df)

# Label encoding categorical columns
le_gender = LabelEncoder()
le_occ = LabelEncoder()
le_bmi = LabelEncoder()
le_sleep = LabelEncoder()

new["Gender"] = le_gender.fit_transform(new["Gender"])
new["Occupation"] = le_occ.fit_transform(new["Occupation"])
new["BMI Category"] = le_bmi.fit_transform(new["BMI Category"])
new["Sleep Disorder"] = le_sleep.fit_transform(new["Sleep Disorder"])

# Blood pressure split
new[["Systolic", "Diastolic"]] = new["Blood Pressure"].str.split("/", expand=True)
new[["Systolic", "Diastolic"]] = new[["Systolic", "Diastolic"]].astype(int)

# features and label
X = new[[
    "Gender","Age","Occupation","Sleep Duration","Quality of Sleep",
    "Physical Activity Level","Stress Level","BMI Category",
    "Systolic","Diastolic","Heart Rate","Daily Steps"
]].values

y = new["Sleep Disorder"].values

# scale features for deep nets
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------
# Define wrappers for 4 DL models
# ---------------------------
epochs_common = 80
batch_common = 32

models_to_run = [
    ("Dense_Small", KerasClassifierWrapper(build_dense_small, epochs=epochs_common, batch_size=batch_common, verbose=0)),
    ("Dense_Deep", KerasClassifierWrapper(build_dense_deep, epochs=epochs_common, batch_size=batch_common, verbose=0)),
    ("Wide_Deep", KerasClassifierWrapper(build_wide_deep, epochs=epochs_common, batch_size=batch_common, verbose=0)),
    ("Dense_BN_Dropout", KerasClassifierWrapper(build_dense_bn_dropout, epochs=epochs_common, batch_size=batch_common, verbose=0)),
]

results = []

# Train & evaluate each model
for name, wrapper in models_to_run:
    print("\n" + "="*8 + f" Training & Evaluating: {name} " + "="*8)
    # fit on training set; we provide validation split for monitoring
    wrapper.fit(X_train, y_train, validation_data=(X_test, y_test))

    # predictions
    y_pred = wrapper.predict(X_test)

    # basic metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision (weighted): {prec*100:.2f}%")
    print(f"Recall (weighted): {rec*100:.2f}%")
    print(f"F1 Score (weighted): {f1*100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ROC AUC (try test probs, else manual CV)
    try:
        auc_val, source = compute_auc_with_fallback_dl(wrapper, X_test, y_test, X, y, cv_folds=5, epochs_cv=40, batch_size_cv=32)
    except Exception as e:
        print("AUC computation raised:", e)
        auc_val, source = None, None

    if auc_val is not None:
        print(f"ROC AUC ({source}): {auc_val*100:.2f}%")
    else:
        print("ROC AUC: could not be computed.")

    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc_val
    })

# Summary
summary = pd.DataFrame(results).set_index("model")
summary["accuracy_pct"] = (summary["accuracy"] * 100).round(2)
summary["precision_pct"] = (summary["precision"] * 100).round(2)
summary["recall_pct"] = (summary["recall"] * 100).round(2)
summary["f1_pct"] = (summary["f1"] * 100).round(2)
summary["roc_auc_pct"] = summary["roc_auc"].apply(lambda x: round(x*100, 2) if x is not None else np.nan)

print("\n\n=== SUMMARY ===")
print(summary[["accuracy_pct","precision_pct","recall_pct","f1_pct","roc_auc_pct"]])

# Optional: Save summary
summary.to_csv("dl_models_sleep_summary.csv")
print("\nSaved 'dl_models_sleep_summary.csv'.")
