# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:47:20 2022

@author: virgi
"""
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import joblib
from math import pi


#API_URL = 'http://127.0.0.1:8000/'
API_URL = 'https://app-myfastapi.herokuapp.com/'
LOGO_IMAGE = "logo.png"
IMAGE_SHAPE = ("summary_plot_shap.png")


#####################################################
#########################################
# Define specific functions
#########################################
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

############################################


@st.cache
def load_data():
    """Function to load the dataframe containing preprocessed data"""
    data = pd.read_csv("data_sampled.csv")
    data = data.dropna(axis=0)
    data = data.sample(n=10000)
    return data


@st.cache
def load_info_customer(customer_id):
    """Function to obtain the raw variables of a unique customer based on its id"""
    full_data = load_data()
    data_customer = full_data.loc[full_data["SK_ID_CURR"] == customer_id]
    return data_customer


@st.cache
def load_data_customer_prepared(customer_id):
    """Function to obtain the encoded and normalized data of a unique customer based on its id"""
    X_prepared_all = pd.read_csv("X_prepared_sampled.csv")
    X_prepared_id = X_prepared_all.loc[X_prepared_all["SK_ID_CURR"] == customer_id]

    return X_prepared_id


@st.cache
def get_prediction(customer_id):
    api_url = 'https://app-myfastapi.herokuapp.com/predict/' + str(customer_id)
    response = requests.get(url=api_url)
    API_data = response.json()
    print('API_DATA', API_data)
    return API_data


@st.cache
def get_neighbors(customer_id):
    api_url = "https://app-myfastapi.herokuapp.com/load_voisins/" + \
        str(customer_id)
    response = requests.get(url=api_url)
    API_knn = response.json()
    print('API_DATA', API_knn)
    return API_knn


@st.cache
def get_shap_values(X_prepared):
    explainer = joblib.load("shap_explainer.joblib")
    shap_values = explainer.shap_values(X_prepared)
    return shap_values


def main():
    """Main function to determine what is display on the dashboard"""

    ##################################################
    # SIDE BAR
    ###################################################

    with st.sidebar:
        st.image(LOGO_IMAGE)
        html_temp = """
                <p style="font-size: 20px; font-weight: bold; text-align:center">
                Page d'accueil</p>
                """
        st.markdown(html_temp, unsafe_allow_html=True)
        #st.xrite("Page d'accueil")

    ##################################################
    # HOME PAGE
    ###################################################

    html_temp = """
                <div style="background-color: gray; padding:10px; border-radius:10px">
                <h1 style="color: white; text-align:center">Bienvenue dans votre application</h1>
                <h1 style="color: white; text-align:center">Prêt à dépenser</h1>
                </div>
                <p style="font-size: 20px; font-weight: bold; text-align:center">
                Support de décision crédit à destination des gestionnaires de la relation client</p>
                """
    st.markdown(html_temp, unsafe_allow_html=True)

    with st.expander(label="🤔 A quoi sert cette application ?"):
        st.write("Ce dashboard interactif de **Prêt à dépenser**\
                 permet de comprendre et interpréter les décisions d'octroi de prêt.\
                 Ces décisions sont la résultante d'une prédictions faites\
                 par un modèle d'apprentissage à partir des données de clients précédents")
        st.text('\n')
        st.write("**Objectif**:  répondre au soucis de transparence vis-à-vis\
                 des décisions d’octroi de crédit qui va tout à fait\
                 dans le sens des valeurs que l’entreprise veut incarner")

    with st.expander(label="🤔 Quels sont les données les plus importantes pour l'octroie d'un crédit ?"):
        st.write("L'octroie d'un crédit à un client est dans un premier temps accordé\
                  via l'utilisation d'un modèle de machine learning de type Light GBM")
        st.text('\n')
        st.write("Pour le modèle, les informations principales utilisées par le modèle sont représentées dans le graphique ci-dessous")
        st.image(IMAGE_SHAPE)


if __name__ == "__main__":
    main()

# to run in a terminal
# streamlit run dashboard.py
# or
# python -m streamlit run myfile.py
