# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:37:36 2022

@author: virgi
"""

# 1- Library imports
import re
import pandas as pd
from lightgbm import LGBMClassifier
from pydantic import BaseModel
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (LabelEncoder,
                                   StandardScaler)
from sklearn.neighbors import NearestNeighbors
from rename_categories import (rename_cat_type_suite,
                               rename_cat_income_type,
                               rename_cat_education_type,
                               rename_cat_family_status,
                               rename_cat_housing_type,
                               rename_organization_type,
                               rename_cat_occupation_type)


class Customers:
    def get_customer_ids():
        """function to get the list of all the customer id of the dataset"""
        data = pd.read_csv("df_final.csv")
        customer_ids = data["SK_ID_CURR"].tolist()
        return customer_ids


class CreditModel:
    """Used for model training and for making predictions"""

    def __init__(self):
        """Class construcot, loads the dataset and loads the model if exists.
        if not, calls the _train_model method and saves the model
        """
        self.model_fname_ = 'model_credit.joblib'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)

    def _train_model(self):
        """Method use to perform the model training with the lgbm algorithm.
        The methods returns the trained model"""

        X = pd.read_csv("X_train_scaled.csv")
        y = pd.read_csv("y_train.csv")
        lgbm = LGBMClassifier(n_estimators=1200,
                              max_depth=4,
                              learning_rate=0.0423,
                              subsample=0.2,
                              colsample_bytree=0.9,
                              num_leaves=92,
                              reg_alpha=0.1,
                              reg_lambda=0.2,
                              class_weight='balanced')
        model = lgbm.fit(X, y)
        return model

    def preprocessor(self, data_customer):
        # Prepare data to predict
        data_to_pp = data_customer.drop(
            ["SK_ID_CURR", "TARGET", "Test"], axis=1)
        data_to_pp = data_to_pp.rename(columns=lambda x:
                                       re.sub('[^A-Za-z0-9_]+', '', x))

        data_to_pp = rename_cat_type_suite(data_to_pp)
        data_to_pp = rename_cat_income_type(data_to_pp)
        data_to_pp = rename_cat_education_type(data_to_pp)
        data_to_pp = rename_cat_family_status(data_to_pp)
        data_to_pp = rename_cat_housing_type(data_to_pp)
        data_to_pp = rename_organization_type(data_to_pp)
        data_to_pp = rename_cat_occupation_type(data_to_pp)

        # Prepare data for fit
        X_train = pd.read_csv("X_train.csv")
        X_train = X_train.drop(columns=["SK_ID_CURR"], axis=1)

        # Encode categorical features
        cat_variables = list(X_train.dtypes[X_train.dtypes == object].index)
        X_train_cat = X_train[cat_variables]
        X_test_cat = data_to_pp[cat_variables]

        le = LabelEncoder()
        count_le = 0

        # Iterate through the columns
        for col in X_train_cat:
            # If 2 or fewer unique categories
            if (X_train_cat.loc[:, col].dtype == 'object'
                    and len(list(X_train_cat.loc[:, col].unique())) <= 2):
                # Train on the training data
                le.fit(X_train_cat.loc[:, col])
                # Transform both testing data
                X_train_cat.loc[:, col] = le.transform(X_train_cat.loc[:, col])
                X_test_cat.loc[:, col] = le.transform(X_test_cat.loc[:, col])
                count_le += 1
        # One-hot encoding for categorical variables with more than 2 classes
        X_train_cat = pd.get_dummies(X_train_cat)
        X_test_cat = pd.get_dummies(X_test_cat)

        # Align data_train and data_test
        X_train_cat, X_test_cat = X_train_cat.align(X_test_cat,
                                                    join="outer",
                                                    axis=1)

        X_test_cat = X_test_cat.fillna(0).astype(int)

        # Numerical variables
        # Create a list with the name of numerical variables
        float_variables = list(X_train.dtypes[X_train.dtypes == float].index)
        int_variables = list(X_train.dtypes[X_train.dtypes == 'int64'].index)
        num_variables = float_variables + int_variables

        # Selection data that to be standardized
        X_train_num = X_train[num_variables]
        X_test_num = data_to_pp[num_variables]

        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train_num)
        X_train_num_i = imputer.transform(X_train_num)

        if sum(X_test_num.isnull().sum()) != 0:
            print('ici')
            X_test_num_i = imputer.transform(X_test_num)
        else:
            print('là')
            X_test_num_i = X_test_num

        scaler = StandardScaler()
        scaler.fit(X_train_num_i)
        X_test_num_scaled = scaler.transform(X_test_num_i)

        # Transform back to pandas Dataframe
        X_test_num_scaled = pd.DataFrame(X_test_num_scaled,
                                         index=X_test_num.index,
                                         columns=X_test_num.columns)

        X_test_scaled = X_test_cat.merge(X_test_num_scaled,
                                         right_index=True,
                                         left_index=True)

        return X_test_scaled

    def predict_target(self, data_customer):
        """
        Method use to make a prediction from input parameters.
        The method retruns the prediction and the prediction probability
        """
        X_prepared = data_customer
        pred_score = self.model.predict(X_prepared)
        pred_proba = self.model.predict_proba(X_prepared)

        return pred_score, pred_proba, X_prepared


def entrainement_knn(df):

    print("En cours...")
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)

    return knn
