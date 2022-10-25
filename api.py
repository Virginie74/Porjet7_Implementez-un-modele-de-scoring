# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:27:26 2022

@author: virgi
"""

# Library imports
from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import json


# Create the app object
app = FastAPI()


def predict_target(data_customer, model):
    """
    Method use to make a prediction from input parameters.
    The method retruns the prediction and the prediction probability
    """
    X_prepared = data_customer
    model = model
    pred_score = model.predict(X_prepared)
    pred_proba = model.predict_proba(X_prepared)

    return pred_score, pred_proba


def entrainement_knn(df):

    print("En cours...")
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)

    return knn

# Index route


@app.get('/')
def index():
    """Function to define the homepage message of the API"""
    return {'message': 'Bienvenue sur l API pret a depenser!'}


@app.get('/predict/{customer_id}')
async def predict(customer_id: int):
    """Function to calculate prediction score and probability for a specific customer"""

    X_all_scaled = pd.read_csv("X_prepared_sampled.csv")
    model = joblib.load('model_credit.joblib')
    data_customer = X_all_scaled.loc[X_all_scaled["SK_ID_CURR"] == customer_id]
    data_customer = data_customer.drop(columns=["SK_ID_CURR"], axis=1)
    pred_score, pred_proba = predict_target(data_customer, model)
    pred_proba = pred_proba[0]
    results = {'pred_score': pred_score[0],
               'pred_proba': pred_proba[1]}

    return results


@app.get('/load_voisins/{customer_id}')
async def load_voisins(customer_id: int):
    X_all_scaled = pd.read_csv("X_prepared_sampled.csv")
    X_all_scaled_prepared = X_all_scaled.drop(columns=["SK_ID_CURR"], axis=1)
    data_customer = X_all_scaled.loc[X_all_scaled["SK_ID_CURR"] == customer_id]
    data_customer = data_customer.drop(columns=["SK_ID_CURR"], axis=1)

    knn = entrainement_knn(X_all_scaled_prepared)
    distances, indices = knn.kneighbors(data_customer)

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_voisins = X_all_scaled.iloc[indices[0], :]

    response = json.loads(df_voisins.to_json(orient='index'))

    return response


# Run the API with uvicorn
# if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8080)

# uvicorn api:app --reload
