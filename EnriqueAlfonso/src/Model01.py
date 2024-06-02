import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import time

start_time = time.time()

df = pd.read_csv('../data/preprocessed_winequality-red.csv')

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

grid_params_rf = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    grid_params_rf,
    verbose=1,
    cv=3,
    n_jobs=-1
)

smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

gs_results_rf = gs_rf.fit(X_train_smote, y_train_smote)

print("Best Parameters from Grid Search (Random Forest): ", gs_results_rf.best_params_)

rf_best = gs_results_rf.best_estimator_
y_pred_rf = rf_best.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_rf)
print(cm)

print(classification_report(y_test, y_pred_rf))

joblib.dump(rf_best, '../models/model_rf.joblib')
joblib.dump(scaler, '../models/scaler.joblib')

end_time = time.time()

print(f"Execution Time: {end_time - start_time} seconds")
