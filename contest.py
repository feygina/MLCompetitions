import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import cross_validation


data = pd.read_csv('x_train.csv', header=None, na_values='?')
y = pd.read_csv('y_train.csv', header=None, na_values='?')


data = pd.concat([data, y], axis=1)
X = pd.DataFrame(data, dtype=float)

X.columns = ['A' + str(i) for i in range(1, len(X.columns))]+['class']
y = X['class']
X.drop(X.columns[[-1]], axis=1, inplace=True)
# print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

model_rfc = ensemble.RandomForestClassifier(n_estimators=100, random_state=123)
model_rfc.fit(X_train, y_train)

err_test = np.mean(y_test != model_rfc.predict(X_test))
print("err ", err_test)

importance = model_rfc.feature_importances_
indices = np.argsort(importance)[::-1]
print(indices)
sum = 0
print("Feature importance:")
for f, idx in enumerate(indices):
    # print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, X.columns[idx], importance[idx]))
    sum += importance[idx]
    # print(f+1)
    # print(sum)

best_features = indices[:3]
best_features_names = X.columns[best_features].values
# print(best_features_names)
best_features_data = data.iloc[:, best_features]
# svm

clf = svm.SVC(C=1)
scores = cross_validation.cross_val_score(clf, best_features_data, y, cv=5)
print("Mean err: ", 100*(1-np.mean(scores)))
