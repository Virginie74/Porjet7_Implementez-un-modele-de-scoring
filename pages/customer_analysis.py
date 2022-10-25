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
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from dashboard_functions import (plot_radar,
                                 rename_columns,
                                 histo_failure,
                                 boxplot_for_num_feature)


API_URL = 'http://127.0.0.1:8000/'
path = 'C:/Users/virgi/OneDrive/Desktop/Projet7_github/Data_dashboard/'
LOGO_IMAGE = "logo.png"

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
def load_data_all_prepared():
    """Function to obtain the encoded and normalized data"""
    X_all_scaled_id = pd.read_csv(path + "X_prepared_sampled.csv")
        
    return X_all_scaled_id


@st.cache
def load_data_customer_prepared(customer_id):
    """Function to obtain the encoded and normalized data of a unique customer based on its id"""
    X_prepared_all = pd.read_csv(path + "X_prepared_sampled.csv")
    X_prepared_id = X_prepared_all.loc[X_prepared_all["SK_ID_CURR"] == customer_id]

    return X_prepared_id


@st.cache
def get_prediction(customer_id):
    api_url = "http://127.0.0.1:8000/predict/"+ str(customer_id)
    response = requests.get(url=api_url)
    API_data = response.json()
    print('API_DATA', API_data)
    return API_data


@st.cache
def get_neighbors(customer_id):
    api_url = "http://127.0.0.1:8000/load_voisins/"+ str(customer_id)
    response = requests.get(url=api_url)
    API_knn = response.json()
    print('API_DATA', API_knn)
    return API_knn


@st.cache
def get_shap_explainer():
    explainer = joblib.load(path + "shap_explainer.joblib")
    return explainer


st.set_page_config(page_title="Dossier du client", page_icon="📈")

with st.spinner('Chargement de la base de données...'):
    data = load_data()
    customer_ids = data["SK_ID_CURR"].tolist()
    X_prepared_all = load_data_all_prepared()

with st.sidebar:
    st.image(LOGO_IMAGE)
    customer_id = st.selectbox("Sélectionner l'identifiant du client",
                                customer_ids)


    customer_info = load_info_customer(customer_id)
    customer_info = rename_columns(customer_info)
    
    personal_info_cols = ["Genre",
                            "Age",
                            "Situation familiale",
                            "Nombre d'enfants",
                            "Niveau d'étude",
                            "Revenu annuel",
                            "Type de crédit demandé",
                            "Montant du crédit",
                            "Taux d'endettement estimé",
                            "Montant des annuités",
                            "Situation professionelle",
                            "Ancienneté dans l'entreprise",
                            "Type de revenu",
                            "Score du client d'après SOURCE 2",
                            "Score du client d'après SOURCE 3",
                            "Propriétaire d'un véhicle",
                            "Propriétaire d'un logement principales"]

    default_list=["Genre",
                    "Age",
                    "Situation familiale",
                    "Nombre d'enfants",
                    "Revenu annuel",
                    "Montant du crédit",
                    "Taux d'endettement estimé"]


    st.write("## Actions à effectuer")
    
    show_credit_decision = st.checkbox("Afficher la décision de crédit")
    show_client_details = st.checkbox("Afficher plus d'informations personelles")
    show_client_comparison = st.checkbox("Comparer aux autres clients")


    if show_credit_decision:
        st.header("Informations personnelles")
        with st.spinner('Chargement des informations personelles du client...'):
            personal_info_df = customer_info[personal_info_cols]
            personal_info_df["Age"] = int(round(personal_info_df["Age"] / -365))
            personal_info_df["Ancienneté dans l'entreprise"] = int(
                    round(personal_info_df["Ancienneté dans l'entreprise"] / -365))

            df_info = personal_info_df[default_list]
            df_info['SK_ID_CURR'] = customer_info['SK_ID_CURR']
            df_info = df_info.set_index('SK_ID_CURR')

            st.table(df_info.astype(str).T)




##################################################
#HOME PAGE
###################################################

html_temp = """
            <div style="background-color: gray; padding:6px; border-radius:10px">
            <h1 style="color: white; text-align:center">Analyse du dossier client</h1>
            </div>
            """
st.markdown(html_temp, unsafe_allow_html=True)
st.write("  ")
st.write("  ")

if show_credit_decision:
    st.write("Identifiant client Sélectionné :", customer_id)

    customer_info = load_info_customer(customer_id)
    customer_info = rename_columns(customer_info)

                
    X_prepared_id = load_data_customer_prepared(customer_id)

    personal_info_cols = ["Genre",
                            "Age",
                            "Situation familiale",
                            "Nombre d'enfants",
                            "Niveau d'étude",
                            "Revenu annuel",
                            "Type de crédit demandé",
                            "Montant du crédit",
                            "Taux d'endettement estimé",
                            "Montant des annuités",
                            "Situation professionelle",
                            "Ancienneté dans l'entreprise",
                            "Type de revenu",
                            "Score du client d'après SOURCE 2",
                            "Score du client d'après SOURCE 3",
                            "Propriétaire d'un véhicle",
                            "Propriétaire d'un logement principales"]


    st.header('‍Décision de crédit')

    with st.spinner('Chargement du score du client...'):
        #Appel de l'API :
        API_data = get_prediction(customer_id)
        classe_predite = API_data['pred_score']
        if classe_predite == 1:
            decision = '❌ Crédit Refusé'
        else:
            decision = '✅ Crédit Accordé'
        proba = API_data['pred_proba']

        client_score = round(proba*100, 2)

        left_column, right_column = st.columns((1, 2))

        left_column.markdown('Risque de non remboursement: **{}%**'.format(str(client_score)))
        left_column.markdown('Seuil de décision: **50%**')

        if classe_predite == 1:
            left_column.markdown(
                'Décision: <span style="color:red">**{}**</span>'.format(decision),\
                unsafe_allow_html=True)   
        else:    
            left_column.markdown(
                'Décision: <span style="color:green">**{}**</span>'\
                .format(decision), \
                unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode = "gauge+delta+number",
            title = {'text': 'Pourcentage de risque de non remboursement'},
            value = client_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'axis': {'range': [None, 100]},
                        'steps' : [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "lightyellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"},
                            ],
                        'threshold': {
                    'line': {'color': "black", 'width': 10},
                    'thickness': 0.8,
                    'value': client_score},

                        'bar': {'color': "black", 'thickness' : 0.2},
                    },
            ))

        gauge.update_layout(width=450, height=250, 
                            margin=dict(l=50, r=50, b=0, t=0, pad=4))

        right_column.plotly_chart(gauge)


    show_local_feature_importance = st.checkbox(
        "Afficher les variables ayant le plus contribuées à la décision du modèle")

    if (show_local_feature_importance):
                number = st.slider('Sélectionner le nombre de features à afficher', \
                                    2, 20, 8)
                with st.spinner('Chargement des données demandées...'):
                    X_prepared_id = X_prepared_id.drop(columns=["SK_ID_CURR"], axis=1)
                    X_prepared_id = rename_columns(X_prepared_id)
                    shap_explainer = get_shap_explainer()
                    shap_values_test = shap_explainer.shap_values(X_prepared_id)
                    
                    fig, ax = plt.subplots(figsize=(15, 15))
                    shap.plots._waterfall.waterfall_legacy(shap_explainer.expected_value[1],
                                                           shap_values_test[1][0],
                                                           feature_names=X_prepared_id.columns,
                                                           max_display=10)
                    st.pyplot(fig)

    show_neighbors = st.checkbox(
        "Afficher les clients similaires")

    if (show_neighbors):
        similar_id = get_neighbors(customer_id)
        df_similar_customer = pd.DataFrame.from_dict(similar_id, orient='index')
        id_similar_customer = df_similar_customer["SK_ID_CURR"].index.tolist()
        df_filtered = data.iloc[id_similar_customer]
        df_filtered = rename_columns(df_filtered)
        df_filtered = df_filtered[["TARGET", "Genre", "Age", "Score du client d'après SOURCE 2", "Score du client d'après SOURCE 3"]]
        print(df_filtered["Age"].dtype)
        df_filtered["Age"] = round(df_filtered["Age"] / -365)
        df_filtered = df_filtered.reset_index(drop=True)
        st.markdown("<u>Liste des 10 dossiers les plus proches de ce client :</u>", unsafe_allow_html=True)
        #st.table(df_filtered.astype(str))           
        st.dataframe(df_filtered.style.highlight_max(axis=0, subset=["TARGET"])) 

if show_client_details:
    st.header("Informations personnelles")
    
    with st.spinner('Chargement des informations personelles du client...'):
        personal_info_df = customer_info[personal_info_cols]
        personal_info_df["Age"] = int(round(personal_info_df["Age"] / -365))
        personal_info_df["Ancienneté dans l'entreprise"] = int(
            round(personal_info_df["Ancienneté dans l'entreprise"] / -365))  

        df_info = personal_info_df
        df_info['SK_ID_CURR'] = customer_info['SK_ID_CURR']
        df_info = df_info.set_index('SK_ID_CURR')

        st.table(df_info.astype(str).T)
        show_all_info = st\
        .checkbox("Afficher toutes les informations (dataframe brute)")
        if (show_all_info):
            st.dataframe(customer_info)


if (show_client_comparison):
    st.header('‍Comparaison avec les autres clients')

    st.markdown("Principales données")

    #Prepare data for radar plot
    data_french_id = rename_columns(X_prepared_id)
    data_french = rename_columns(X_prepared_all)
    df_MM = data_french_id[["Age", "Montant des annuités", "Durée du crédit",
                            "Ancienneté dans l'entreprise", "Montant de l'achat", "Revenu annuel"]]
    df_normal = data_french[["Age", "Montant des annuités", "Durée du crédit",
                            "Ancienneté dans l'entreprise", "Montant de l'achat", "Revenu annuel"]]

    df_repaid = data.loc[data["TARGET"] == 0]
    id_repaid = df_repaid["SK_ID_CURR"]
    data_french_repaid = pd.merge(id_repaid, X_prepared_all, how='inner', on=["SK_ID_CURR"])
    data_french_repaid = rename_columns(data_french_repaid)

    df_mean_repaid = data_french_repaid[["Age", "Montant des annuités", "Durée du crédit",
                                    "Ancienneté dans l'entreprise", "Montant de l'achat", "Revenu annuel"]]
    df_default = data.loc[data["TARGET"] == 1]
    id_default = df_default["SK_ID_CURR"]
    data_french_default = pd.merge(id_default, X_prepared_all, how='inner', on=["SK_ID_CURR"])
    data_french_default = rename_columns(data_french_default)
    
    df_mean_default = data_french_default[["Age", "Montant des annuités", "Durée du crédit",
                            "Ancienneté dans l'entreprise", "Montant de l'achat", "Revenu annuel"]]

    for column in df_MM.columns:
        df_MM[column] = (df_MM[column] - df_normal[column].min()) / (df_normal[column].max() - df_normal[column].min())  
        df_mean_repaid[column] = (df_mean_repaid[column] - df_normal[column].min()) / (df_normal[column].max() - df_normal[column].min())
        df_mean_default[column] = (df_mean_default[column] - df_normal[column].min()) / (df_normal[column].max() - df_normal[column].min())
    
    df_mean_repaid = df_mean_repaid.mean()
    df_mean_default = df_mean_default.mean()


    fig = plot_radar(df_MM, df_mean_repaid, df_mean_default)

    # Show the graph
    st.pyplot(fig)

    
    if ("SK_ID_CURR" in data.columns) & ("TARGET" in data.columns):
        data_selec = data.drop(columns=["SK_ID_CURR", "TARGET"], axis=1)
    else:
        data_selec = data
    data_selec = rename_columns(data_selec)
    data_comp = rename_columns(data)
    feature = st.selectbox("Sélectionner une variable",\
                        list(data_selec.columns))
    

    if (data_comp[feature].dtype == 'int64') | (data_comp[feature].dtype == float):
        fig = boxplot_for_num_feature(data_comp, feature, customer_info)
    if data_comp[feature].dtype == object:
        if len(data_comp[feature].unique()) < 4:
            fig = histo_failure(data_comp, feature, customer_info)
        else:
            fig = histo_failure(data_comp, feature, customer_info, label_rotation=True)
    st.pyplot(fig)
