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
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('train.csv', header=None, na_values='?')

data.columns = ['A' + str(i) for i in range(1, len(data.columns))]+['class']

y = data['class']
data.drop(data.columns[[-1]], axis=1, inplace=True)
data = pd.DataFrame(data, dtype=float)


def divide_test_train(df_binary):
    test_df = df_binary.loc[df_binary['Better'] == 0]
    train_df = df_binary.loc[df_binary['Better'] != 0]
    return test_df, train_df
#
# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=123)
# model_rfc = ensemble.RandomForestClassifier(n_estimators=100, random_state=123)
# model_rfc.fit(X_train, y_train)
#
# importance = model_rfc.feature_importances_
# indices = np.argsort(importance)[::-1]



# for i in range(2, 31):
#     best_features = indices[:i]
#     best_features_data = data.iloc[:, best_features]
#     bdt = ensemble.RandomForestClassifier(n_estimators=300, random_state=123)
#     # bdt.fit(data, y)
#     scores = cross_validation.cross_val_score(bdt, best_features_data, y, cv=2)
#     print("AdaBoost: ", 100*(np.mean(scores)))

# X_train, X_test, y_train, y_test = train_test_split(best_features_data, y, test_size=0.5, random_state=123)
# kmen = KMeans(n_clusters=3, max_iter=300, random_state=123)
# for i in range(16, 17):
#     best_features = indices[:i]
#     best_features_data = data.iloc[:, best_features]
#     X_train, X_test, y_train, y_test = train_test_split(best_features_data, y, test_size=0.5, random_state=123)
#     x_new_test = X_test[:]
#     x_new_train = X_train[:]
#     kmen.fit(X_train, y_train)
#     res1 = kmen.predict(X_test)
#     x_new_test['kmean'] = pd.Series(res1, index=x_new_test.index)
#     # print(x_new_test)
#     err_test = np.mean(y_test == res1)
#     print("err ", err_test)
#     kmen.fit(X_test, y_test)
#     res2 = kmen.predict(X_train)
#     x_new_train['kmean'] = pd.Series(res2, index=x_new_train.index)
#     err_test = np.mean(y_train == res2)
#     # print(x_new_train)
#     print("err ", err_test)
#
# new_data = pd.concat([x_new_test, x_new_train])


# bdt = ensemble.RandomForestClassifier(n_estimators=700, random_state=123)
# # bdt.fit(data, y)
# scores = cross_validation.cross_val_score(bdt, new_data, y, cv=2)
# print("AdaBoost: ", 100*(np.mean(scores)))


#
# x_train = data.as_matrix()
yy_train = y.as_matrix()

# for i in range(10, 20):
#     best_features = indices[:i]
#     best_features_data = data.iloc[:, best_features]
#     new_data = best_features_data.as_matrix()
#     # gbm.fit(x_train, y_train)
#     for j in np.arange(0.01, 0.11, 0.01):
#         for k in range(100, 1100, 100):
#             gbm = xgb.XGBClassifier(max_depth=3, n_estimators=k, learning_rate=j)
#             scores = cross_validation.cross_val_score(gbm, new_data, y_train, cv=2)
#             print("XGB: ", "data ", i, "rate ", j, "est ", k,  100*(np.mean(scores)))
# predictions = gbm.predict(x_test)
# predictions = pd.DataFrame(predictions, dtype=float)
#
# predictions.to_csv('results.csv', sep=',', mode='w')

# best_features = indices[:26]
#
#
# test = pd.read_csv('x_test.csv', header=None, na_values='?')
# test.columns = ['A' + str(i) for i in range(1, len(test.columns)+1)]
# test = pd.DataFrame(test, dtype=float)
# best_test = test.iloc[:, best_features]
# best_test_ever = best_test.as_matrix()
#
#
# best_features_data = data.iloc[:, best_features]
# new_data = best_features_data.as_matrix()
# gbm = xgb.XGBClassifier(max_depth=3, n_estimators=600, learning_rate=0.07)
# scores = cross_validation.cross_val_score(gbm, new_data, y_train, cv=2)
# print("XGB: ",  100*(np.mean(scores)))
# gbm.fit(new_data, yy_train)
# res = gbm.predict(best_test_ever)
# print(res)
