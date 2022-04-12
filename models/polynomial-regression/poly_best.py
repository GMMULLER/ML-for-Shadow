'''

Using Linear Regression model with Polynomial Features of degree 4. 

Independent variables:
heightroof_sum, groundelev_sum, exposure

Dependent variable:
june21

Using k-fold cross validation with k=10.

R2 = 0.5535105029662832

'''

import psycopg2
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

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

columns_names = ['heightroof_sum', 'groundelev_sum', 'exposure', 'june21']    

x_end = len(columns_names)-1

df = load_data(columns_names)

X = df.iloc[:,0:x_end].values
Y = df.iloc[:,x_end:len(columns_names)].values

# reescalling X values to [0,1]
scale = MinMaxScaler()
X = scale.fit_transform(X)

# Parameters
_polynomial_features_degree = 4 

poly = PolynomialFeatures(degree=_polynomial_features_degree)

model = LinearRegression()

pipeline = Pipeline([("polynomial_features", poly), ("linear_regression", model)])

scores = cross_val_score(pipeline, X, Y, scoring="r2", cv=10)

print("R2: "+str(scores.mean()))