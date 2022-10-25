# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:47:20 2022

@author: virgi
"""
import os
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import joblib
from math import pi
from dashboard_functions import (plot_radar,
                                 rename_columns,
                                 histo_failure,
                                 boxplot_for_num_feature)


API_URL = 'http://127.0.0.1:8000/'
path = 'C:/Users/virgi/OneDrive/Desktop/Projet7_github/Data_dashboard/'
LOGO_IMAGE = "logo.png"
IMAGE_SHAPE = ("summary_plot_shap.png")


@st.cache
def load_data():
    """Function to load the dataframe containing preprocessed data"""
    data = pd.read_csv(path + "data_sampled.csv")
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
    X_prepared_all = pd.read_csv(path + "X_prepared_sampled.csv")
    X_prepared_id = X_prepared_all.loc[X_prepared_all["SK_ID_CURR"] == customer_id]

    return X_prepared_id


@st.cache
def get_prediction(customer_id):
    api_url = "http://127.0.0.1:8000/predict/" + str(customer_id)
    response = requests.get(url=api_url)
    API_data = response.json()
    print('API_DATA', API_data)
    return API_data


@st.cache
def get_neighbors(customer_id):
    api_url = "http://127.0.0.1:8000/load_voisins/" + str(customer_id)
    response = requests.get(url=api_url)
    API_knn = response.json()
    print('API_DATA', API_knn)
    return API_knn


@st.cache
def get_shap_values(X_prepared):
    explainer = joblib.load(path + "shap_explainer.joblib")
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
