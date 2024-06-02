
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd
import time

start_time = time.time()

df = pd.read_csv('../data/preprocessed_winequality-red.csv')

X = df.drop('quality', axis=1)
y = df['quality']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    grid_params = {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 17, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    gs = GridSearchCV(
        KNeighborsClassifier(),
        grid_params,
        verbose=1,
        cv=3,
        n_jobs=-1
    )

    gs_results = gs.fit(X_train_scaled, y_train)

    print("Best Parameters from Grid Search: ", gs_results.best_params_)

smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

knn = KNeighborsClassifier(n_neighbors=gs_results.best_params_['n_neighbors'], weights=gs_results.best_params_['weights'], metric=gs_results.best_params_['metric'])
knn.fit(X_train_smote, y_train_smote)

y_pred = knn.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

joblib.dump(knn, '../models/model.joblib')

end_time = time.time()

print(f"Execution Time: {end_time - start_time} seconds")
