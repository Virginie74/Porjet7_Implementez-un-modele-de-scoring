import matplotlib.pyplot as plt
import matplotlib
import random
import seaborn as sns
import pandas as pd
import numpy as np
import gc

def rename_cat_type_suite(df_app, variable="NAME_TYPE_SUITE"):
    cat = {"Unaccompanied": "Unaccompanied",
           "Family": "Accompanied",
           "Spouse, partner": "Accompanied",
           "Children": "Accompanied",
           "Other_B": "Accompanied",
           "Other_A": "Accompanied",
           "Group of people": "Accompanied"}
    df_app = df_app.replace({variable: cat})

    return (df_app)


def rename_cat_income_type(df_app, variable="NAME_INCOME_TYPE"):
    cat = {"Working": "Working",
           "State servant": "Working",
           "Commercial associate": "Working",
           "Pensioner": "Pensioner",
           "Unemployed": "Unemployed",
           "Student": "Unemployed",
           "Businessman": "Working",
           "Maternity leave": "Unemployed"}
    df_app = df_app.replace({variable: cat})

    return (df_app)


def rename_cat_education_type(df_app, variable="NAME_EDUCATION_TYPE"):
    cat = {"Academic degree": "Higher_education",
           "Higher education": "Higher_education",
           "Secondary / secondary special": "Secondary_education",
           "Incomplete higher": "Secondary_education",
           "Lower secondary": "Lower_secondary",
           "Secondary education":"Secondary_education"}

    df_app = df_app.replace({variable: cat})

    return (df_app)


def rename_cat_family_status(df_app, variable="NAME_FAMILY_STATUS"):
    cat = {"Married": "Married",
           "Civil marriage": "Married",
           "Single / not married": "Single_and_related",
           "Separated": "Single_and_related",
           "Widow": "Single_and_related"}

    df_app = df_app.replace({variable: cat})

    return (df_app)


def rename_cat_housing_type(df_app, variable="NAME_HOUSING_TYPE"):
    cat = {"With parents": "Social_or_equivalent_housing",
           "Municipal apartment": "Social_or_equivalent_housing",
           "Office apartment": "Social_or_equivalent_housing",
           "Co-op apartment": "Social_or_equivalent_housing",
           "Rented apartment": "House_or_apartment",
           "House / apartment": "House_or_apartment"}

    df_app = df_app.replace({variable: cat})

    return (df_app)


def rename_organization_type(df_app, variable="ORGANIZATION_TYPE"):
    cat = {'Business Entity Type 3': 'Business_Entity',
           'School': 'School',
           'Government': 'Government',
           'Religion': 'Other',
           'Other': 'Other',
           'XNA': 'Other',
           'Electricity': 'Construction',
           'Medicine': 'Services',
           'Business Entity Type 2': 'Business_Entity',
           'Self-employed': 'Services',
           'Transport: type 2': 'Transport',
           'Construction': 'Construction',
           'Housing': 'Construction',
           'Kindergarten': 'School',
           'Trade: type 7': 'Trade',
           'Industry: type 11': 'Industry',
           'Military': 'Government',
           'Services': 'Services',
           'Security Ministries': 'Government',
           'Transport: type 4': 'Transport',
           'Industry: type 1': 'Industry',
           'Emergency': 'Government',
           'Security': 'Government',
           'Trade: type 2': 'Trade',
           'University': 'School',
           'Transport: type 3': 'Transport',
           'Police': 'Government',
           'Business Entity Type 1': 'Business_Entity',
           'Postal': 'Government',
           'Industry: type 4': 'Industry',
           'Agriculture': 'Other',
           'Restaurant': 'Services',
           'Culture': 'Services',
           'Hotel': 'Services',
           'Industry: type 7': 'Industry',
           'Trade: type 3': 'Trade',
           'Industry: type 3': 'Industry',
           'Bank': 'Services',
           'Industry: type 9': 'Industry',
           'Insurance': 'Services',
           'Trade: type 6': 'Trade',
           'Industry: type 2': 'Industry',
           'Transport: type 1': 'Transport',
           'Industry: type 12': 'Industry',
           'Mobile': 'Services',
           'Trade: type 1': 'Trade',
           'Industry: type 5': 'Industry',
           'Industry: type 10': 'Industry',
           'Legal Services': 'Government',
           'Advertising': 'Services',
           'Trade: type 5': 'Trade',
           'Cleaning': 'Services',
           'Industry: type 13': 'Industry',
           'Trade: type 4': 'Trade',
           'Telecom': 'Construction',
           'Industry: type 8': 'Industry',
           'Realtor': 'Services',
           'Industry: type 6': 'Industry'}

    df_app = df_app.replace({variable: cat})

    return (df_app)


def rename_cat_occupation_type(df_app, variable="OCCUPATION_TYPE"):
    cat = {"Sales staff": "Sales_staff",
           "Core staff": "Core_staff",
           "High skill tech staff": "IT_staff",
           "Medicine staff": "Medicine_staff",
           "Security staff": "Security_staff",
           "Cooking staff": "Restaurant_bar_staff",
           "Cleaning staff": "Cleaning_staff",
           "Private service staff": "Private_service_staff",
           "Low-skill laborers": "Laborers",
           "Waiters/barmen staff": "Restaurant_bar_staff",
           "Realty agents": "Core_staff",
           "HR staff": "Core_staff",
           "IT staff": "IT_staff",
           "Secretaries": "Core_staff"}

    df_app = df_app.replace({variable: cat})

    return (df_app)
