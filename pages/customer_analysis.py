import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import shap
from math import pi
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir))


#API_URL = 'http://127.0.0.1:8000/'
API_URL = 'https://app-myfastapi.herokuapp.com/'
LOGO_IMAGE = "logo.png"

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
def load_data_all_prepared():
    """Function to obtain the encoded and normalized data"""
    X_all_scaled_id = pd.read_csv("X_prepared_sampled.csv")

    return X_all_scaled_id


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
    api_url = 'https://app-myfastapi.herokuapp.com/load_voisins/' + \
        str(customer_id)
    response = requests.get(url=api_url)
    API_knn = response.json()
    print('API_DATA', API_knn)
    return API_knn


@st.cache
def get_shap_explainer():
    explainer = joblib.load("shap_explainer.joblib")
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

    default_list = ["Genre",
                    "Age",
                    "Situation familiale",
                    "Nombre d'enfants",
                    "Revenu annuel",
                    "Montant du crédit",
                    "Taux d'endettement estimé"]

    st.write("## Actions à effectuer")

    show_credit_decision = st.checkbox("Afficher la décision de crédit")
    show_client_details = st.checkbox(
        "Afficher plus d'informations personelles")
    show_client_comparison = st.checkbox("Comparer aux autres clients")

    if show_credit_decision:
        st.header("Informations personnelles")
        with st.spinner('Chargement des informations personelles du client...'):
            personal_info_df = customer_info[personal_info_cols]
            personal_info_df["Age"] = int(
                round(personal_info_df["Age"] / -365))
            personal_info_df["Ancienneté dans l'entreprise"] = int(
                round(personal_info_df["Ancienneté dans l'entreprise"] / -365))

            df_info = personal_info_df[default_list]
            df_info['SK_ID_CURR'] = customer_info['SK_ID_CURR']
            df_info = df_info.set_index('SK_ID_CURR')

            st.table(df_info.astype(str).T)


##################################################
# HOME PAGE
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
        # Appel de l'API :
        API_data = get_prediction(customer_id)
        classe_predite = API_data['pred_score']
        if classe_predite == 1:
            decision = '❌ Crédit Refusé'
        else:
            decision = '✅ Crédit Accordé'
        proba = API_data['pred_proba']

        client_score = round(proba*100, 2)

        left_column, right_column = st.columns((1, 2))

        left_column.markdown(
            'Risque de non remboursement: **{}%**'.format(str(client_score)))
        left_column.markdown('Seuil de décision: **50%**')

        if classe_predite == 1:
            left_column.markdown(
                'Décision: <span style="color:red">**{}**</span>'.format(
                    decision),
                unsafe_allow_html=True)
        else:
            left_column.markdown(
                'Décision: <span style="color:green">**{}**</span>'
                .format(decision),
                unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+delta+number",
            title={'text': 'Pourcentage de risque de non remboursement'},
            value=client_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]},
                   'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "lightyellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"},
            ],
                'threshold': {
                'line': {'color': "black", 'width': 10},
                    'thickness': 0.8,
                    'value': client_score},

                'bar': {'color': "black", 'thickness': 0.2},
            },
        ))

        gauge.update_layout(width=450, height=250,
                            margin=dict(l=50, r=50, b=0, t=0, pad=4))

        right_column.plotly_chart(gauge)

    show_local_feature_importance = st.checkbox(
        "Afficher les variables ayant le plus contribuées à la décision du modèle")

    if (show_local_feature_importance):
        number = st.slider('Sélectionner le nombre de features à afficher',
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
        df_similar_customer = pd.DataFrame.from_dict(
            similar_id, orient='index')
        id_similar_customer = df_similar_customer["SK_ID_CURR"].index.tolist()
        df_filtered = data.iloc[id_similar_customer]
        df_filtered = rename_columns(df_filtered)
        df_filtered = df_filtered[[
            "TARGET", "Genre", "Age", "Score du client d'après SOURCE 2", "Score du client d'après SOURCE 3"]]
        print(df_filtered["Age"].dtype)
        df_filtered["Age"] = round(df_filtered["Age"] / -365)
        df_filtered = df_filtered.reset_index(drop=True)
        st.markdown(
            "<u>Liste des 10 dossiers les plus proches de ce client :</u>", unsafe_allow_html=True)
        # st.table(df_filtered.astype(str))
        st.dataframe(df_filtered.style.highlight_max(
            axis=0, subset=["TARGET"]))

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

    # Prepare data for radar plot
    X_prepared_id = load_data_customer_prepared(customer_id)
    X_prepared_id = X_prepared_id.drop(columns=["SK_ID_CURR"], axis=1)
    data_french_id = rename_columns(X_prepared_id)
    data_french = rename_columns(X_prepared_all)
    df_MM = data_french_id[["Age", "Montant des annuités", "Durée du crédit",
                            "Ancienneté dans l'entreprise", "Montant de l'achat", "Revenu annuel"]]
    df_normal = data_french[["Age", "Montant des annuités", "Durée du crédit",
                            "Ancienneté dans l'entreprise", "Montant de l'achat", "Revenu annuel"]]

    df_repaid = data.loc[data["TARGET"] == 0]
    id_repaid = df_repaid["SK_ID_CURR"]
    data_french_repaid = pd.merge(
        id_repaid, X_prepared_all, how='inner', on=["SK_ID_CURR"])
    data_french_repaid = rename_columns(data_french_repaid)

    df_mean_repaid = data_french_repaid[["Age", "Montant des annuités", "Durée du crédit",
                                         "Ancienneté dans l'entreprise", "Montant de l'achat", "Revenu annuel"]]
    df_default = data.loc[data["TARGET"] == 1]
    id_default = df_default["SK_ID_CURR"]
    data_french_default = pd.merge(
        id_default, X_prepared_all, how='inner', on=["SK_ID_CURR"])
    data_french_default = rename_columns(data_french_default)

    df_mean_default = data_french_default[["Age", "Montant des annuités", "Durée du crédit",
                                           "Ancienneté dans l'entreprise", "Montant de l'achat", "Revenu annuel"]]

    for column in df_MM.columns:
        df_MM[column] = (df_MM[column] - df_normal[column].min()) / \
            (df_normal[column].max() - df_normal[column].min())
        df_mean_repaid[column] = (df_mean_repaid[column] - df_normal[column].min()) / (
            df_normal[column].max() - df_normal[column].min())
        df_mean_default[column] = (df_mean_default[column] - df_normal[column].min()) / (
            df_normal[column].max() - df_normal[column].min())

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
    feature = st.selectbox("Sélectionner une variable",
                           list(data_selec.columns))

    if (data_comp[feature].dtype == 'int64') | (data_comp[feature].dtype == float):
        fig = boxplot_for_num_feature(data_comp, feature, customer_info)
    if data_comp[feature].dtype == object:
        if len(data_comp[feature].unique()) < 4:
            fig = histo_failure(data_comp, feature, customer_info)
        else:
            fig = histo_failure(data_comp, feature,
                                customer_info, label_rotation=True)
    st.pyplot(fig)
