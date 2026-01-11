# sleep_models_compare.py
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# CLASSIFIERS (sklearn)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# ---------------------------
# Helper: robust AUC computation (same idea as your function)
# ---------------------------
def compute_auc_with_fallback(model, X_test, y_test, X_all, y_all, cv_fallback=5):
    """
    Try to compute AUC using model.predict_proba on X_test.
    If it fails or test lacks classes, compute cross-validated predict_proba on full data.
    Returns (roc_auc, source_string) or (None, None) on failure.
    """
    classes = getattr(model, "classes_", None)
    # Try to get classes from y_all if model hasn't been fitted with classes_
    if classes is None:
        classes = np.unique(y_all)

    unique_test_classes = np.unique(y_test)

    # Try predict_proba on the model (works if model was fitted and supports predict_proba)
    y_prob_test = None
    try:
        y_prob_test = model.predict_proba(X_test)
    except Exception:
        y_prob_test = None

    # If probabilities exist and at least two classes present in test
    if y_prob_test is not None and unique_test_classes.size >= 2:
        try:
            if classes.shape[0] == 2:
                # binary classification => pick positive class column
                # find index of the positive class (commonly 1); fallback to column 1
                pos_classes = list(classes)
                if 1 in pos_classes:
                    pos_index = pos_classes.index(1)
                else:
                    pos_index = 1 if y_prob_test.shape[1] > 1 else 0
                roc_auc = roc_auc_score(y_test, y_prob_test[:, pos_index])
            else:
                # multiclass
                y_test_bin = label_binarize(y_test, classes=classes)
                roc_auc = roc_auc_score(y_test_bin, y_prob_test, average='weighted', multi_class='ovr')
            return roc_auc, "test_proba"
        except Exception:
            pass

    # Fallback: cross-validated probabilities on full dataset
    try:
        cv_probs = cross_val_predict(model, X_all, y_all, cv=cv_fallback, method='predict_proba')
        # Determine classes from model after cross_val_predict (it uses clones, so classes might be from model)
        classes_cv = np.unique(y_all)
        if classes_cv.shape[0] == 2:
            pos_classes = list(classes_cv)
            if 1 in pos_classes:
                pos_index = pos_classes.index(1)
            else:
                pos_index = 1 if cv_probs.shape[1] > 1 else 0
            roc_auc = roc_auc_score(y_all, cv_probs[:, pos_index])
        else:
            y_all_bin = label_binarize(y_all, classes=classes_cv)
            roc_auc = roc_auc_score(y_all_bin, cv_probs, average='weighted', multi_class='ovr')
        return roc_auc, f"cv{cv_fallback}"
    except Exception:
        return None, None

# ---------------------------
# Load and preprocess (mirrors your original code)
# ---------------------------
# NOTE: change the path to your file as needed
df = pd.read_csv(r"D:\AA\Projects\sleep analysis\sleep.csv")
new = pd.DataFrame(df)

# Label encoders - keep them for decoding later if needed
le_gender = LabelEncoder()
le_occ = LabelEncoder()
le_bmi = LabelEncoder()
le_sleep = LabelEncoder()

new["Gender"] = le_gender.fit_transform(new["Gender"])
new["Occupation"] = le_occ.fit_transform(new["Occupation"])
new["BMI Category"] = le_bmi.fit_transform(new["BMI Category"])
new["Sleep Disorder"] = le_sleep.fit_transform(new["Sleep Disorder"])

# Blood pressure split (assumes format "systolic/diastolic")
new[["Systolic", "Diastolic"]] = new["Blood Pressure"].str.split("/", expand=True)
new[["Systolic", "Diastolic"]] = new[["Systolic", "Diastolic"]].astype(int)

# Features and label
inputs = new[[
    "Gender","Age","Occupation","Sleep Duration","Quality of Sleep",
    "Physical Activity Level","Stress Level","BMI Category",
    "Systolic","Diastolic","Heart Rate","Daily Steps"
]]

output = new["Sleep Disorder"]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    inputs, output, test_size=0.2, random_state=42, stratify=output
)

# ---------------------------
# Classifiers list
# ---------------------------
models = [
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ("SVC", SVC(probability=True, random_state=42)),  # probability=True to enable predict_proba
    ("KNeighbors", KNeighborsClassifier(n_neighbors=5)),
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=42)),
    ("GaussianNB", GaussianNB()),
    ("ExtraTrees", ExtraTreesClassifier(n_estimators=200, random_state=42)),
    ("MLP", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
]

# Storage for results
results = []

# ---------------------------
# Train, predict, evaluate each model
# ---------------------------
for name, model in models:
    print(f"\n{'='*8} Training & Evaluating: {name} {'='*8}")
    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Basic metrics
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

    # ROC AUC using helper (may fallback to cv)
    roc_auc_value, source = compute_auc_with_fallback(model, X_test, y_test, inputs, output, cv_fallback=5)
    if roc_auc_value is not None:
        print(f"ROC AUC ({source}): {roc_auc_value*100:.2f}%")
    else:
        print("ROC AUC: could not be computed for this model.")

    # Save
    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc_value
    })

# ---------------------------
# Summary DataFrame
# ---------------------------
summary = pd.DataFrame(results).set_index("model")
# convert AUC to percent for readability and keep NaNs for models where AUC couldn't be computed
summary["accuracy_pct"] = (summary["accuracy"] * 100).round(2)
summary["precision_pct"] = (summary["precision"] * 100).round(2)
summary["recall_pct"] = (summary["recall"] * 100).round(2)
summary["f1_pct"] = (summary["f1"] * 100).round(2)
summary["roc_auc_pct"] = summary["roc_auc"].apply(lambda x: round(x*100, 2) if x is not None else np.nan)

print("\n\n=== Summary of all models ===")
print(summary[["accuracy_pct","precision_pct","recall_pct","f1_pct","roc_auc_pct"]])

# Optional: Save summary to CSV
summary.to_csv("sleep_models_comparison_summary.csv")
print("\nSaved summary to 'sleep_models_comparison_summary.csv' (in current working directory).")

# ---------------------------
# Example: predicting user input using a chosen model
# ---------------------------
# If you want a quick interactive prediction using a chosen model (e.g., RandomForest),
# uncomment and use the block below. Make sure you enter encoded categorical values
# consistent with training encoding (or use the LabelEncoders to transform).
#
# chosen_model = models[1][1]  # RandomForest instance (already trained above)
# try:
#     gender = int(input("enter your gender (encoded): "))
#     age = int(input("enter your age: "))
#     occupation = int(input("enter your occupation (encoded): "))
#     sleepdu = float(input("enter your sleep duration: "))
#     sleepqu = float(input("enter your sleep quality: "))
#     phycialadct = float(input("enter your physical activity level: "))
#     StressLevel = float(input("enter your stress level: "))
#     BMICategory = int(input("enter your BMI Category (encoded): "))
#     Systolic = float(input("enter your Systolic: "))
#     Diastolic = float(input("enter your Diastolic: "))
#     heartrate = float(input("enter your heart rate: "))
#     dailystep = float(input("enter your daily steps: "))
# except Exception as e:
#     print("Invalid input:", e)
#     raise
#
# user_feat = np.array([[gender, age, occupation, sleepdu, sleepqu, phycialadct,
#                        StressLevel, BMICategory, Systolic, Diastolic, heartrate, dailystep]])
# pred = chosen_model.predict(user_feat)
# decoded = le_sleep.inverse_transform(pred.astype(int))[0]
# print("Predicted label:", decoded)
