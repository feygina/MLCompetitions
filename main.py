import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import cross_validation


plt.style.use('ggplot')

data = pd.read_csv('crx_data_train_x.csv', header=None, na_values='?')
y = pd.read_csv('crx_data_train_y.csv', header=None, na_values='?')


data = pd.merge(data, y, on=0)
data = pd.DataFrame(data)
data.drop(data.columns[[0]], axis=1, inplace=True)

data.columns = ['A' + str(i) for i in range(1, len(data.columns))]+['class']
# print(data.head())

# print(data.count(axis=0))
data = data.fillna(data.median(axis=0), axis=0)
# print(data.count(axis=0))

# print(data.describe(include=[object]))
data_describe = data.describe(include=[object])
categorical_columns = data.describe(include=[object]).columns.values
numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']
for c in categorical_columns:
    data[c] = data[c].fillna(data_describe[c]['top'])
# print(data.describe(include=[object]))
# print(data.count(axis=0))

binary_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
for c in binary_columns:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1

data_nonbinary = pd.get_dummies(data[nonbinary_columns])
# print(data[numerical_columns].describe())
data_numerical = data[numerical_columns]
data_numerical_normal_1 = (data_numerical - data_numerical.mean()) / data_numerical.std()
# print(data_numerical_normal_1.describe())
data_numerical_normal_2 = (data_numerical - data_numerical.mean()) / (data_numerical.max()-data_numerical.min())
# print(data_numerical_normal_2.describe())

data = pd.concat((data_numerical_normal_1, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)
# print(data.columns.values)

X = data.drop('class', axis=1)
y = data['class']

kfold = 5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

model_rfc = ensemble.RandomForestClassifier(n_estimators=100, random_state=123)
model_rfc.fit(X_train, y_train)

err_test = np.mean(y_test != model_rfc.predict(X_test))
# print(err_test)

importance = model_rfc.feature_importances_
indices = np.argsort(importance)[::-1]
# print(indices)
# print("Feature importance:")
# for f, idx in enumerate(indices):
#     print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, X.columns[idx], importance[idx]))

best_features = indices[:8]
best_features_names = X.columns[best_features].values
# print(best_features_names)
best_features_data = data.iloc[:, best_features]
# print(best_features_data.head())

X_train, X_test, y_train, y_test = train_test_split(best_features_data, y, test_size=0.25, random_state=123)

# scores = cross_validation.cross_val_score(model_rfc, data, y, cv=kfold)
# print(scores)
# feature selection

# svm

clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, best_features_data, y, cv=10)
print("Mean err: ", 100*(1-np.mean(scores)))
