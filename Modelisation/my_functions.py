import matplotlib.pyplot as plt
import matplotlib
import random
import seaborn as sns
import pandas as pd
import numpy as np
import gc


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()
               * 100).sort_values(ascending=False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


def missing_data_var(data, var):
    nb_var = data[var].isnull().sum()
    percent = (nb_var/data.shape[0]*100)
    return percent


def list_color(n=2):
    dict_colors = matplotlib.colors.cnames
    list_to_suppress = ['gainsboro', 'whitesmoke', 'white', 'snow', 'mistyrose', 'seashell', 'linen', 'anitquewhite',
                        'oldlace', 'floralwhite', 'cornsilk', 'ivory', 'beige', 'lightyellow', 'lightgoldenrodyellow',
                        'honeydew', 'mintcream', 'azure', 'lightcyan', 'aliceblue', 'ghostwhite', 'lavender', 'thistle',
                        'lavenderblush']

    mydict = {k: v for k, v in dict_colors.items() if k not in list_to_suppress}

    color_selected = random.choices(list(mydict.items()), k=n)
    color = []
    for i in range(len(color_selected)):
        color.append(color_selected[i][0])

    return color


# Faire une représentation graphique
def hist_distrib_target(df_app, variable="YEARS_BIRTH", bins=100):
    # Initiate figure
    fig = plt.figure(figsize=(10, 10))
    plt.gcf().subplots_adjust(left=0.3,
                              bottom=0.3,
                              right=3,
                              top=1,
                              wspace=0.5,
                              hspace=15)

    # Plot the histogram
    ax1 = fig.add_subplot(1, 3, 1)
    ax1 = sns.histplot(x=variable,
                       data=df_app,
                       bins=bins,
                       color="dimgrey")
    ax1.set_xlabel("{var}".format(var=variable))
    ax1.set_ylabel("Nombre de clients")
    ax1.set_title("{var}".format(var=variable))
    sns.despine()

    # Plot the kdeplot as a function of "TARGET"
    ax2 = fig.add_subplot(1, 3, 2)
    ax2 = sns.kdeplot(df_app.loc[df_app["TARGET"] == 0, variable],
                      label="target == 0", color="darkturquoise")
    ax2 = sns.kdeplot(df_app.loc[df_app["TARGET"] == 1, variable],
                      label="target == 1", color="tomato")
    ax2.legend(labels=["target == 0", "target == 1"],
               bbox_to_anchor=(0.05, 0.95), loc='upper left', title='TARGET')
    ax2.set_xlabel("{var}".format(var=variable))
    ax2.set_ylabel("Density")
    sns.despine()

    # Calculate and write correlation coeff with TARGET
    corr = df_app[variable].corr(df_app['TARGET'])
    ax2.set_title("{var} - Pearson coeff.: {pearson}".format(var=variable,
                                                             pearson=corr))

    ax3 = fig.add_subplot(1, 3, 3)
    ax3 = sns.boxplot(data=df_app,
                      x="TARGET",
                      y=variable,
                      palette=["darkturquoise", "tomato"])
    ax3.set_ylabel("{var}".format(var=variable))

    plt.show()


def hist_distrib(df_app, variable="YEARS_BIRTH", bins=100):
    # Initiate figure
    fig = plt.figure(figsize=(10, 6))

    # Plot the histogram
    ax1 = fig.add_subplot(1, 2, 1)
    ax1 = sns.histplot(x=variable,
                       data=df_app,
                       bins=bins,
                       color="dimgrey")
    ax1.set_xlabel("{var}".format(var=variable))
    ax1.set_ylabel("Nombre de clients")
    ax1.set_title("{var}".format(var=variable))
    sns.despine()

    ax3 = fig.add_subplot(1, 2, 2)
    ax3 = sns.boxplot(data=df_app,
                      y=variable,
                      color="dimgrey")
    ax3.set_ylabel("{var}".format(var=variable))

    plt.show()


def failure_to_repay_graph(df, variable="YEARS_BIRTH", bins=np.linspace(20, 70, num=11)):

    df = df[["TARGET", variable]]

    # Create age bins
    df["variable_BINNED"] = pd.cut(df[variable],
                                   bins=bins)
    # Group the data of the same age bin
    var_groups = df.groupby("variable_BINNED").mean()

    # Visualize the failure rate by bins
    plt.figure(figsize=(8, 8))
    plt.bar(var_groups.index.astype(str), 100 *
            var_groups['TARGET'], color="tomato")
    plt.xticks(rotation=75)
    plt.xlabel('{variable} Group'.format(variable=variable))
    plt.ylabel('Failure to Repay (%)')
    plt.title('Failure to Repay by {variable} Group'.format(variable=variable))
    plt.show()


def hist_distrib_cat(df_app, feature, label_rotation=False, horizontal_layout=True):
    temp = df_app[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,
                       'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df_app[[feature, 'TARGET']].groupby([feature],
                                                   as_index=False).mean()
    cat_perc["TARGET"] = cat_perc["TARGET"]*100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x=feature, y="Number of contracts", data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    s = sns.barplot(ax=ax2,
                    x=feature,
                    y='TARGET',
                    order=cat_perc[feature],
                    data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Failure to Repay (%)', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show()


def hist_distrib_cat_no_target(df_app, feature, label_rotation=False):
    temp = df_app[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,
                       'Number of contracts': temp.values})

    fig = plt.figure()
    sns.set_color_codes("pastel")
    s = sns.barplot(x=feature, y="Number of contracts", data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    plt.show()


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


# Function to calculate missing values by column# Funct


def missing_values_table(df, print_info=False):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    if print_info:
        # Print some summary information
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
              "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def remove_missing_columns(df, threshold=90):
    # Calculate missing stats for train and test (remember to calculate a percent!)
    train_miss = pd.DataFrame(df.isnull().sum())
    train_miss['percent'] = 100 * train_miss[0] / len(df)

    # list of missing columns for train and test
    missing_train_columns = list(
        train_miss.index[train_miss['percent'] > threshold])
    missing_columns = list(set(missing_train_columns))

    # Print information
    print('There are %d columns with greater than %d%% missing values.' %
          (len(missing_columns), threshold))

    # Drop the missing columns and return
    df = df.drop(columns=missing_columns)

    return df

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
