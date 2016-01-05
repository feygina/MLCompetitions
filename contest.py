import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import cross_validation


X = pd.read_csv('x_train.csv', header=None, na_values='?')
y = pd.read_csv('y_train.csv', header=None, na_values='?')

X.columns = ['A' + str(i) for i in range(1, len(X.columns)+1)]
y.columns = ['class']
y = y['class']
# print(X.head())

print(X.count(axis=0))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

model_rfc = ensemble.RandomForestClassifier(n_estimators=100, random_state=123)
model_rfc.fit(X_train, y_train)

err_test = np.mean(y_test != model_rfc.predict(X_test))
print("err ", err_test)

importance = model_rfc.feature_importances_
indices = np.argsort(importance)[::-1]
# print(indices)
# print("Feature importance:")
# for f, idx in enumerate(indices):
#     print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, X.columns[idx], importance[idx]))


# svm

# clf = svm.SVC(C=1)
# scores = cross_validation.cross_val_score(clf, X, y, cv=5)
# print("Mean err: ", 100*(1-np.mean(scores)))
