import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB
import xgboost as xgb
from sklearn.cluster import KMeans


data = pd.read_csv('train.csv', header=None, na_values='?')
test = pd.read_csv('x_test.csv', header=None, na_values='?')

data = pd.DataFrame(data, dtype=float)
test = pd.DataFrame(test, dtype=float)

data.columns = ['A' + str(i) for i in range(1, len(data.columns))]+['class']
test.columns = ['A' + str(i) for i in range(1, len(data.columns))]

y = data['class']
data.drop(data.columns[[-1]], axis=1, inplace=True)


# print(data.duplicated())


X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.3, random_state=123)
model_rfc = ensemble.RandomForestClassifier(n_estimators=100, random_state=123)
model_rfc.fit(X_train, Y_train)

importance = model_rfc.feature_importances_
indices = np.argsort(importance)[::-1]

best_features = indices[:16]

best_test = test.iloc[:, best_features]
best_data = data.iloc[:, best_features]

x_train = best_data.as_matrix()
y_train = y.as_matrix()
x_test = best_test.as_matrix()

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=600, learning_rate=0.07)
scores = cross_validation.cross_val_score(gbm, x_train, y_train, cv=10)
print("XGB: ",  100*(np.mean(scores)))
gbm.fit(x_train, y_train)
res = gbm.predict(best_test)
print(res)

# np.savetxt("foo.csv", res, delimiter=",")
