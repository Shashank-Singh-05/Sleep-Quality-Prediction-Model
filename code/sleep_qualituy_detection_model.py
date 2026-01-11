import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, label_binarize

# LOAD DATA
files = pd.read_csv(r"D:\AA\Projects\sleep analysis\sleep.csv") 
new = pd.DataFrame(files)

# LABEL ENCODING
le_gender = LabelEncoder()
le_occ = LabelEncoder()
le_bmi = LabelEncoder()
le_sleep = LabelEncoder()

new["Gender"] = le_gender.fit_transform(new["Gender"])
new["Occupation"] = le_occ.fit_transform(new["Occupation"])
new["BMI Category"] = le_bmi.fit_transform(new["BMI Category"])
new["Sleep Disorder"] = le_sleep.fit_transform(new["Sleep Disorder"])

# BLOOD PRESSURE SPLIT
new[["Systolic", "Diastolic"]] = new["Blood Pressure"].str.split("/", expand=True)
new[["Systolic", "Diastolic"]] = new[["Systolic", "Diastolic"]].astype(int)

# INPUTS AND OUTPUTS
inputs = new[[
    "Gender","Age","Occupation","Sleep Duration","Quality of Sleep",
    "Physical Activity Level","Stress Level","BMI Category",
    "Systolic","Diastolic","Heart Rate","Daily Steps"
]]

output = new["Sleep Disorder"]

# TRAIN TEST SPLIT
# Use stratify to keep class proportions in train/test; helps AUC computation stability.
X_train, X_test, y_train, y_test = train_test_split(
    inputs, output, test_size=0.2, random_state=42, stratify=output
)

# MODEL TRAINING
algo = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
algo.fit(X_train, y_train)

# PREDICTIONS
y_pred = algo.predict(X_test)

# BASIC METRICS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\nModel Accuracy : {accuracy * 100:.2f}%")
print(f"Model Precision : {precision * 100:.2f}%")
print(f"Model F1 Score : {f1 * 100:.2f}%")
print(f"Model Recall : {recall * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ROC AUC (robust)
def compute_auc_with_fallback(model, X_test, y_test, X_all, y_all, cv_fallback=5):
    """
    Try to compute AUC using model.predict_proba on X_test.
    If it fails (e.g. missing classes in y_test), compute cross-validated probabilities on full data.
    """
    classes = model.classes_
    unique_test_classes = np.unique(y_test)

    # If predict_proba available, try it first
    try:
        y_prob_test = model.predict_proba(X_test)
    except Exception as e:
        y_prob_test = None
        # print the reason (not fatal)
        print("predict_proba() not available on model/prediction:", e)

    # If we have probabilities and at least two classes present in test, compute AUC
    if y_prob_test is not None and unique_test_classes.size >= 2:
        try:
            if classes.shape[0] == 2:
                # Binary classification: pick probability for positive class (class at index 1)
                # Ensure we pick the correct column by using classes order.
                pos_index = 1  # by convention, second column
                # If classes aren't [0,1], still take column 1 (XGBoost returns columns matching classes_)
                roc_auc = roc_auc_score(y_test, y_prob_test[:, pos_index])
            else:
                y_test_bin = label_binarize(y_test, classes=classes)
                roc_auc = roc_auc_score(y_test_bin, y_prob_test,
                                        average='weighted', multi_class='ovr')
            return roc_auc, "test_proba"
        except Exception as e:
            print("AUC on test probabilities failed:", e)

    # Fallback: compute cross-validated probabilities on full dataset
    print("Falling back to cross-validated predict_proba (this may be slower)...")
    try:
        cv_probs = cross_val_predict(model, X_all, y_all, cv=cv_fallback, method='predict_proba')
        classes_cv = model.classes_
        if classes_cv.shape[0] == 2:
            pos_index = 1
            roc_auc = roc_auc_score(y_all, cv_probs[:, pos_index])
        else:
            y_all_bin = label_binarize(y_all, classes=classes_cv)
            roc_auc = roc_auc_score(y_all_bin, cv_probs, average='weighted', multi_class='ovr')
        return roc_auc, f"cv{cv_fallback}"
    except Exception as e:
        print("Cross-validated AUC computation failed:", e)
        return None, None

roc_auc_value, source = compute_auc_with_fallback(algo, X_test, y_test, inputs, output, cv_fallback=5)
if roc_auc_value is not None:
    print(f"\nModel ROC AUC ({source}) : {roc_auc_value * 100:.2f}%")
else:
    print("\nModel ROC AUC : could not be computed. Consider checking classes / using more data or different CV.")

# USER INPUT PREDICTION 
# NOTE: categorical inputs must be encoded the same way as training data.
# We'll ask for encoded values (int). If you want to accept raw text, use the corresponding encoder.transform([text]).
try:
    gender = int(input("enter your gender: "))
    age = int(input("enter your age: "))
    occupation = int(input("enter your Occupation: "))
    sleepdu = float(input("enter your sleep duration: "))
    sleepqu = float(input("enter your sleep quality: "))
    phycialadct = float(input("enter your physical activity level: "))
    StressLevel = float(input("enter your Stress Level: "))
    BMICategory = int(input("enter your BMI Category: "))
    Systolic = float(input("enter your Systolic: "))
    Diastolic = float(input("enter your Diastolic: "))
    heartrate = float(input("enter your heart rate: "))
    dailystep = float(input("enter your daily steps: "))
except Exception as e:
    print("Invalid input:", e)
    raise

result = algo.predict([[gender, age, occupation, sleepdu, sleepqu,
                        phycialadct, StressLevel, BMICategory,
                        Systolic, Diastolic, heartrate, dailystep]])

# decode predicted label back to original string label
pred_label = le_sleep.inverse_transform(result.astype(int))[0]

# FINAL RESULT
if result == 2:
    print("Perfect: Good Lifestyle")
elif result == 1:
    print("Sorry: You Have Sleep Apnea")
elif result == 0:
    print("Sorry: You Have Insomnia")
else:
    print(f"Predicted label (decoded): {pred_label}")
