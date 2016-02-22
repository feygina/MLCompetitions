import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import cross_validation


data = pd.read_csv('data/data.csv', header=0, na_values='?', index_col=0, low_memory=False)

# print(data.head())
# number_of_leads = data['is_lead'].value_counts()

y = data['is_lead']
y.fillna(0, inplace=True)
y.replace('lead', 1, inplace=True)

# print(y.value_counts())

data.drop(data.columns[[-1]], axis=1, inplace=True)

# print(data.isnull().sum())

data.dropna(axis=1, how='any', inplace=True)

# data.drop('visit_first_action_time', axis=1, inplace=True)
# data.drop('visit_last_action_time', axis=1, inplace=True)
# data.drop('server_time', axis=1, inplace=True)
# data.drop('visitor_localtime', axis=1, inplace=True)
# data.drop('idvisitor.1', axis=1, inplace=True)
# # cuz it all = False
# data.drop('config_gears', axis=1, inplace=True)
# data.drop('idaction_name_ref', axis=1, inplace=True)
# data.drop('utm_term', axis=1, inplace=True)
# data.drop('referer_url', axis=1, inplace=True)
# data.drop('referer_keyword', axis=1, inplace=True)
# data.drop('utm_content', axis=1, inplace=True)
# data.drop('channel', axis=1, inplace=True)
# data.drop('location_city', axis=1, inplace=True)
# data.drop('config_browser_name', axis=1, inplace=True)
# data.drop('config_browser_version', axis=1, inplace=True)
# data.drop('location_city', axic=1, inplace=True)
# data.drop('referer_name', axis=1, inplace=True)

data['idvisitor'] = data['idvisitor'].apply(id)

list_of_columns_names = list(data.columns)
list_of_count_columns_names = ['visitor_count_visits', 'visitor_days_since_last', 'visitor_days_since_first',
                               'visit_total_actions', 'visit_total_time', 'time_spent_ref_action']

count_data = data[list_of_count_columns_names]

# data.drop(list_of_count_columns_names, axis=1, inplace=True)

# print(data['channel'].value_counts())


# train_df = data.loc[data['is_lead'] == 'lead']
# predict_df = data.loc[data['is_lead'] != 'lead']

# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=123)
# model_rfc = ensemble.RandomForestClassifier(n_estimators=100, random_state=123)
# model_rfc.fit(X_train, y_train)
#
# importance = model_rfc.feature_importances_
# indices = np.argsort(importance)[::-1]
# print(indices)
