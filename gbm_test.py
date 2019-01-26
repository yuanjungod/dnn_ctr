# -*- coding:utf-8 -*-

from utils import data_preprocess
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

result_dict = data_preprocess.read_criteo_data('./data/tiny_train_input.csv', './data/category_emb.csv')
test_dict = data_preprocess.read_criteo_data('./data/tiny_test_input.csv', './data/category_emb.csv')

X_train, y_train = result_dict['index'], result_dict['label']
X_test, y_test = test_dict['index'], test_dict['label']

gbm = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100)
gbm.fit(np.array(X_train), np.array(y_train), eval_set=[(np.array(X_test), np.array(y_test))],
        eval_metric='l1', early_stopping_rounds=5)
print(sum(y_train)/len(y_train))


print("train auc:", roc_auc_score(y_train, gbm.predict_proba(X_train)[:, 1]))
print("test auc:", roc_auc_score(y_test, gbm.predict_proba(X_test)[:, 1]))



