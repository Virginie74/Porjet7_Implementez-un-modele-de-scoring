# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 11:27:26 2022

@author: virgi
"""

# Library imports
import uvicorn
from fastapi import FastAPI, HTTPException
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from starlette.responses import JSONResponse
import json
from api_functions import (Customers,
                           CreditModel,
                           entrainement_knn)

# Create the app object
app = FastAPI()


path = 'C:/Users/virgi/OneDrive/Desktop/Projet7_github/Data_dashboard/'

# Import the model
model = CreditModel()


# Index route
@app.get('/')
def index():
    """Function to define the homepage message of the API"""
    return {'message': 'Bienvenue sur l API pret a depenser!'}


@app.get('/customers')
async def get_customers():
    """Function to get a list of the all the customer ids"""
    customer_ids = Customers.get_customer_ids()
    if len(customer_ids) == 0:
        raise HTTPException(status_code=418, detail="Something went wrong\
                            ...No customer in the database")
    return {"customer_id possible": [customer_ids]}


@app.get('/customers/{cutomer_id}')
async def check_customer(customer_id: int):
    """function to verify the presence of an id customer in the dataframe"""
    customer_ids = Customers.get_customer_ids()
    if customer_id not in customer_ids:
        raise HTTPException(status_code=404, detail="Customer ID not found")
    return {"customer_id": [customer_id]}


async def load_data():
    """Function to load the dataframe containing preprocessed data"""
    full_data = pd.read_csv(path + "df_sampled.csv")
    return full_data


@app.get('/predict/{customer_id}')
async def predict(customer_id: int):
    """Function to calculate prediction score and probability for a specific customer"""

    X_all_scaled = pd.read_csv(path + "X_prepared_sampled.csv")
    data_customer = X_all_scaled.loc[X_all_scaled["SK_ID_CURR"] == customer_id]
    data_customer = data_customer.drop(columns=["SK_ID_CURR"], axis=1)
    pred_score, pred_proba, X_prepared = model.predict_target(data_customer)
    pred_proba = pred_proba[0]
    results = {'pred_score': pred_score[0],
               'pred_proba': pred_proba[1]}

    return results


@app.get('/load_voisins/{customer_id}')
async def load_voisins(customer_id: int):
    X_all_scaled = pd.read_csv(path + "X_prepared_sampled.csv")
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
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn api:app --reload
