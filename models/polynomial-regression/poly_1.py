import matplotlib.pyplot as plt
import psycopg2
import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from itertools import combinations

def load_data(columns):

    #Establishing the connection
    conn = psycopg2.connect(database="tccbase", user='postgres', password='admin', host='127.0.0.1', port= '5432')

    #Setting auto commit false
    conn.autocommit = True

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    where_clause = ""
    select_columns = ""

    d = {} # Used to build the dataframe

    for key,column in enumerate(columns):
        d[column] = []

        if(key != len(columns)-1):
            select_columns += column+','
            where_clause += column+" IS NOT NULL AND "
        else:
            select_columns += column
            where_clause += column+" IS NOT NULL"

    sql = "SELECT "+select_columns+" FROM complete_sky_exposure WHERE "+where_clause+";"

    cursor.execute(sql)

    resultado = cursor.fetchall()

    for row in resultado:
        for key,column in enumerate(columns):
            d[column].append(float(row[key]))                

    df = pd.DataFrame(data=d) # Generating dataframe

    return df

r_means = []

for i in [4]:

    columns_names = ['heightroof_sum', 'groundelev_sum', 'exposure', 'june21']    

    print("Calculando i: "+str(i)+"...")

    x_end = len(columns_names)-1

    df = load_data(columns_names)

    X = df.iloc[:,0:x_end].values
    Y = df.iloc[:,x_end:len(columns_names)].values

    # reescalling X values to [0,1]
    scale = MinMaxScaler()
    X = scale.fit_transform(X)

    # Parameters
    _polynomial_features_degree = i # Up to five the variation is considerable

    poly = PolynomialFeatures(degree=_polynomial_features_degree)

    model = LinearRegression()

    pipeline = Pipeline([("polynomial_features", poly), ("linear_regression", model)])

    scores = cross_val_score(pipeline, X, Y, scoring="neg_root_mean_squared_error", cv=5)

    r_means.append(scores.mean())

print(r_means)