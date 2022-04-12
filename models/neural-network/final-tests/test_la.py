from os import pipe
import psycopg2
import pandas as pd
import numpy as np
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error

def load_data(columns):

    #Establishing the connection
    conn = psycopg2.connect(database="tccbase_la", user='postgres', password='admin', host='127.0.0.1', port= '5432')

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

    sql = "SELECT "+select_columns+" FROM var_radial WHERE sep22 IS NOT NULL ORDER BY june21;"

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

def mean_nrmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse/np.mean(y_true.astype(np.float64))

def maxmin_nrmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse/(np.max(y_true.astype(np.float64))-np.min(y_true.astype(np.float64)))

def sd_nrmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse/np.std(y_true.astype(np.float64))

def iq_nrmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse/(np.quantile(y_true.astype(np.float64), 0.75) - np.quantile(y_true.astype(np.float64), 0.25))

columns = ['heightroof_count', 'heightroof_max', 'heightroof_mean']

columns_names = []

for k in [10,40,70,100]:
    for i in [2,4,6,8,10,12,14,16,18,20,22,24]:
        for toAdd in columns:
            columns_names.append(toAdd+"_"+str(k)+"_"+str(i))

columns_names.append("sep22")

x_end = len(columns_names)-1 

df = load_data(columns_names) 

df = df.replace(np.nan, 0)

df = df.sample(frac=1, random_state=1).reset_index(drop=True)

subSetSize = 1

end = int(len(df) * subSetSize)

df = df.iloc[:end, :]

X = df.iloc[:,0:x_end].values 

y = df.iloc[:,x_end:len(columns_names)].values 
y = np.ravel(y) # Flattening the array 

print("Data loaded")

model_file = 'trained_nn_dc_sep22.sav'
scale_file = 'scale_dc_sep22.sav'

loaded_model = joblib.load(model_file)
loaded_scale = joblib.load(scale_file)

X = loaded_scale.transform(X)

y_pred = loaded_model.predict(X)

print("R2", r2_score(y, y_pred))
print("MAPE", mean_absolute_error(y, y_pred))
print("RMSE", mean_squared_error(y, y_pred, squared=False))
print("Mean_NRMSE", mean_nrmse(y, y_pred))
print("MaxMin_NRMSE", maxmin_nrmse(y, y_pred))
print("Std_NRMSE", sd_nrmse(y, y_pred))
print("Iq_NRMSE", iq_nrmse(y, y_pred))
