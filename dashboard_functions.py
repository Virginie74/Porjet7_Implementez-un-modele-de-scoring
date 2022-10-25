# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:24:12 2022

@author: virgi
"""
import re
import requests
from math import pi
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (LabelEncoder,
                                   StandardScaler)
from rename_categories import (rename_cat_type_suite,
                               rename_cat_income_type,
                               rename_cat_education_type,
                               rename_cat_family_status,
                               rename_cat_housing_type,
                               rename_organization_type,
                               rename_cat_occupation_type)

API_URL = 'http://127.0.0.1:8000'


def preprocessor(data_customer):
    """Functions to prepare data customer for prediction in the same way than
    during the model training"""

    # Prepare data to predict
    data_to_pp = data_customer.drop(["SK_ID_CURR", "TARGET", "Test"], axis=1)
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

    encoder = LabelEncoder()
    count_le = 0

    # Iterate through the columns
    for col in X_train_cat:
        # If 2 or fewer unique categories
        if (X_train_cat.loc[:, col].dtype == 'object'
                and len(list(X_train_cat.loc[:, col].unique())) <= 2):
            # Train on the training data
            encoder.fit(X_train_cat.loc[:, col])
            # Transform both testing data
            X_train_cat.loc[:, col] = encoder.transform(
                X_train_cat.loc[:, col])
            X_test_cat.loc[:, col] = encoder.transform(X_test_cat.loc[:, col])
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
    X_test_num_i = imputer.transform(X_test_num)

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


def lime_explainer(customer_id):
    """Function to get the feature importance using lime method"""
    api_url = API_URL + '/lime'
    data_json = {'customer_id': customer_id}
    response = requests.get(api_url, params=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}")
    return response.json()


def rename_columns(df):
    new_name_cols = {
        'CODE_GENDER': "Genre",
        'DAYS_BIRTH': "Age",
        'NAME_FAMILY_STATUS': "Situation familiale",
        'CNT_CHILDREN': "Nombre d'enfants",
        'NAME_EDUCATION_TYPE': "Niveau d'étude",
        'AMT_INCOME_TOTAL': "Revenu annuel",
        'NAME_CONTRACT_TYPE': "Type de crédit demandé",
        'AMT_CREDIT': "Montant du crédit",
        'ANNUITY_INCOME_PERCENT': "Taux d'endettement estimé",
        'AMT_ANNUITY': "Montant des annuités",
        'OCCUPATION_TYPE': "Situation professionelle",
        'DAYS_EMPLOYED': "Ancienneté dans l'entreprise",
        'NAME_INCOME_TYPE': "Type de revenu",
        'EXT_SOURCE_2': "Score du client d'après SOURCE 2",
        'EXT_SOURCE_3': "Score du client d'après SOURCE 3",
        'FLAG_OWN_CAR': "Propriétaire d'un véhicle",
        'FLAG_OWN_REALTY': "Propriétaire d'un logement principales",
        'CREDIT_TERM': "Durée du crédit",
        'DAYS_INSTALMENT_delay': "Délai de remboursement de crédit précédent",
        'DAYS_INSTALMENT_delta': "Delta entre sommes percues et du de crédit précédent",
        'SUM_OF_CURRENT_CREDIT': "Montant des crédits en cours",
        'AMT_GOODS_PRICE': "Montant de l'achat",
        'NB_APPROVED_APP_previous': "Nb de demandes approuvées",
        'NB_REFUSED_APP_previous': "Nb de demandes refusées",
        'REGION_RATING_CLIENT_W_CITY': "Zone d'habitation (commune)",
        'NAME_EDUCATION_TYPE_Higher_education': "Niveau d'éducation universitaire",
        'AMT_REQ_CREDIT_BUREAU_QRT': "Nb de demandes de renseignements",
        'DAYS_LAST_PHONE_CHANGE': "Ancienneté du téléphone (en jours)",
        'OCCUPATION_TYPE_Core_staff': "Activité professionnelle (Personnel clé)",
        'ORGANIZATION_TYPE_School': "Secteur professionel (école)",
        'NB_CLOSED_CREDIT_bureau': "Nb de crédits terminés",
        'NB_ACTIVE_CREDIT_bureau': "Nb de crédits en cours"}
    df.rename(columns=new_name_cols, inplace=True)
    return df


def plot_radar(df_MM, df_mean_repaid, df_mean_default):

    radar_df = pd.DataFrame({"Groupe": ["Client",
                                        "Moyenne crédit remboursé",
                                        "Moyenne défaut de paiement"],
                            "Durée des crédits": [df_MM["Durée du crédit"].values[0],
                                                  df_mean_repaid["Durée du crédit"].mean(
                            ),
        df_mean_default["Durée du crédit"].mean()],
        "Montant des annuités": [1 - df_MM["Montant des annuités"].values[0],
                                 1 -
                                 df_mean_repaid["Montant des annuités"].mean(
        ),
        1 - df_mean_default["Montant des annuités"].mean()],
        "Ancienneté dans l'entreprise": [1 - df_MM["Ancienneté dans l'entreprise"].values[0],
                                         1 -
                                         df_mean_repaid["Ancienneté dans l'entreprise"].mean(
        ),
        1 - df_mean_default["Ancienneté dans l'entreprise"].mean()],
        "Revenu annuel": [df_MM["Revenu annuel"].values[0]*10,
                          df_mean_repaid["Revenu annuel"].mean(
        )*10,
        df_mean_default["Revenu annuel"].mean()*10],
        "Age": [df_MM["Age"].values[0],
                df_mean_repaid["Age"].mean(),
                df_mean_default["Age"].mean()],
        "Montant de l'achat": [df_MM["Montant de l'achat"].values[0],
                               df_mean_repaid["Montant de l'achat"].mean(
        ),
        df_mean_default["Montant de l'achat"].mean()]})

    # Figure initialization
    fig = plt.figure(figsize=(6, 6))

    # Number of variable for radar plot
    N = len(list(radar_df)[1:])

    # Prepare values
    values = radar_df.loc[0].drop("Groupe").values.flatten().tolist()
    values += values[:1]

    # Define angleq
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], list(radar_df)[1:], color='grey', size=8)
    ax.tick_params(axis='x', which='major', pad=40)

    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=7)
    plt.ylim(0, 1)

    # Ind1
    values = radar_df.loc[0].drop("Groupe").values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='#929591', linewidth=1,
            linestyle='solid', label="Client")
    ax.fill(angles, values, color='#929591', alpha=0.4)

    # Ind2
    values = radar_df.loc[1].drop("Groupe").values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='#40E0D0', linewidth=1,
            linestyle='solid', label="Moyenne crédit accordé")
    ax.fill(angles, values, color='#40E0D0', alpha=0.1)

    # Ind2
    values = radar_df.loc[2].drop("Groupe").values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='#FF6347', linewidth=1,
            linestyle='solid', label="Moyenne crédit refusé")
    ax.fill(angles, values, color='#FF6347', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(2, 1))

    ax.tick_params(axis='x', labelsize=12)

    # Show the graph
    return fig


def boxplot_for_num_feature(df, feature, data_customer):
    # Prepare data
    data = df[[feature, "TARGET"]]
    data["TARGET"].replace({0: "Accordés", 1: "Refusé"}, inplace=True)
    fig = plt.figure(figsize=(5, 7))
    ax = sns.boxplot(data=data, x="TARGET", y=feature,
                     palette=["darkturquoise", "tomato"])
    ax = sns.stripplot(data=data, x="TARGET", y=feature, alpha=0.6)
    ax.hlines(y=data_customer[feature].values, xmin=-1, xmax=2,
              color='#ff3300', linestyle='--', linewidth=3, label='Valeur du client')
    plt.legend(bbox_to_anchor=(0.31, 1.06), loc=2, borderaxespad=0.,
               framealpha=1, facecolor='white', frameon=True)
    ax.set_xlabel(" ")
    return fig


def histo_failure(df, feature, data_customer, label_rotation=False, horizontal_layout=True):
    temp = df[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,
                       'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df[[feature, 'TARGET']].groupby([feature],
                                               as_index=False).mean()
    cat_perc["TARGET"] = cat_perc["TARGET"]*100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    if(horizontal_layout):
        fig, ax2 = plt.subplots(figsize=(8, 6))
    else:
        fig, ax2 = plt.subplots(figsize=(12, 14))

    s = sns.barplot(ax=ax2,
                    x=feature,
                    y='TARGET',
                    order=cat_perc[feature],
                    data=cat_perc,
                    palette="pastel")
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Défaut de rembouserment (%)', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylim(0, 20)
    plt.title(f"Catégorie du client: {data_customer[feature].values[0]}")

    return fig
