from os import pipe
import psycopg2
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures

def load_data(columns):

    #Establishing the connection
    conn = psycopg2.connect(database="tccbase_boston", user='postgres', password='admin', host='127.0.0.1', port= '5432')

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

    # sql = "SELECT "+select_columns+" FROM complete_radial_buffer_sky_exposure_1 as c1, complete_radial_buffer_sky_exposure_2 as c2 WHERE "+where_clause+" AND c1.id = c2.id;"
    sql = "SELECT "+select_columns+" FROM var_radial WHERE dec21 IS NOT NULL AND sep22 IS NOT NULL ORDER BY june21;"

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

# columns_names = ['exposure']
columns_names = []

for k in [10,40,70,100]:
    # for i in range(1,25):
    for i in [2,4,6,8,10,12,14,16,18,20,22,24]:
        for toAdd in columns:
            columns_names.append(toAdd+"_"+str(k)+"_"+str(i))

columns_names.append("dec21") #Treina
columns_names.append("sep22") #Testa

x_end = len(columns_names)-2

df = load_data(columns_names) 

df = df.replace(np.nan, 0)

#Randomizando o dataset para pegar somente uma parcela deste para agilizar o treino
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

subSetSize = 1

end = int(len(df) * subSetSize)

df = df.iloc[:end, :]

X = df.iloc[:,0:x_end].values 

scale = StandardScaler() 

bottomIndex = 0
increment = int(len(df)/3)
topIndex = increment

_scores = {
    "R2": [],
    "MAPE": [],
    "RMSE": [],
    "Mean_NRMSE": [],
    "MaxMin_NRMSE": [],
    "Std_NRMSE": [],
    "Iq_NRMSE": []
}

while(True):

    test = df.iloc[bottomIndex:topIndex,:]

    train_1 = df.iloc[:bottomIndex,:]
    train_2 = df.iloc[topIndex:,:]

    train = pd.concat([train_1,train_2])

    X_train = train.iloc[:,1:x_end].values
    X_train = scale.fit_transform(X_train)

    X_test = test.iloc[:,1:x_end].values
    X_test = scale.transform(X_test)

    y_train = train.iloc[:,x_end:len(columns_names)-1].values
    y_train = np.ravel(y_train)

    y_test = test.iloc[:,x_end+1:len(columns_names)].values
    y_test = np.ravel(y_test)

    regr = MLPRegressor(hidden_layer_sizes=(16,), activation='logistic', random_state=1, max_iter=500, verbose=True, solver='adam')

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    
    _scores["R2"].append(r2_score(y_test, y_pred))
    _scores["MAPE"].append(mean_absolute_error(y_test, y_pred))
    _scores["RMSE"].append(mean_squared_error(y_test, y_pred, squared=False))
    _scores["Mean_NRMSE"].append(mean_nrmse(y_test, y_pred))
    _scores["MaxMin_NRMSE"].append(maxmin_nrmse(y_test, y_pred))
    _scores["Std_NRMSE"].append(sd_nrmse(y_test, y_pred))
    _scores["Iq_NRMSE"].append(iq_nrmse(y_test, y_pred))

    if(topIndex == len(df)):
        break

    bottomIndex += increment

    topIndex += increment
    # se o proximo incremento vai alem do maximo
    if topIndex + increment > len(df):
        topIndex = len(df) # Inclui o resto das linhas

print(_scores)

for key in _scores:
    print(key,sum(_scores[key])/len(_scores[key]))