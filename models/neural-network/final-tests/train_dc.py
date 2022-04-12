from os import pipe
import psycopg2
import pandas as pd
import numpy as np
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures

def load_data(columns):

    #Establishing the connection
    conn = psycopg2.connect(database="tccbase_dc", user='postgres', password='admin', host='127.0.0.1', port= '5432')

    #Setting auto commit false
    conn.autocommit = True

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    select_columns = ""

    d = {} # Used to build the dataframe

    for key,column in enumerate(columns):
        d[column] = []

        if(key != len(columns)-1):
            select_columns += column+','
        else:
            select_columns += column

    sql = "SELECT "+select_columns+" FROM var_radial WHERE june21 IS NOT NULL ORDER BY june21;"

    cursor.execute(sql)

    resultado = cursor.fetchall()

    for row in resultado:
        for key,column in enumerate(columns):
            if(row[key]):
                d[column].append(float(row[key]))                
            else:
                d[column].append(row[key])                

    df = pd.DataFrame(data=d) # Generating dataframe

    return df

columns = ['heightroof_count', 'heightroof_max', 'heightroof_mean']

columns_names = []

for k in [10,40,70,100]:
    for i in [2,4,6,8,10,12,14,16,18,20,22,24]:
        for toAdd in columns:
            columns_names.append(toAdd+"_"+str(k)+"_"+str(i))

columns_names.append("june21")

x_end = len(columns_names)-1 

df = load_data(columns_names) 

df = df.replace(np.nan, 0)

df = df.sample(frac=1, random_state=1).reset_index(drop=True)

subSetSize = 1

end = int(len(df) * subSetSize)

df = df.iloc[:end, :]

X = df.iloc[:,0:x_end].values 

scale = StandardScaler() 

y = df.iloc[:,x_end:len(columns_names)].values 
y = np.ravel(y) # Flattening the array 

# X_train = scale.fit_transform(X_train)
# X_test = scale.transform(X_test)

print("Data loaded")

regr = MLPRegressor(hidden_layer_sizes=(16,), activation='logistic', random_state=1, max_iter=500, verbose=True, solver='adam')

X = scale.fit_transform(X)

regr.fit(X, y)

model_file = 'trained_nn_dc_june21.sav'
scale_file = 'scale_dc_june21.sav'

joblib.dump(regr, model_file)
joblib.dump(scale, scale_file)
