from os import pipe
import psycopg2
import pandas as pd
import numpy as np

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
    sql = "SELECT "+select_columns+" FROM var_radial WHERE dec21 IS NOT NULL ORDER BY june21;"

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

columns_names.append("dec21")

x_end = len(columns_names)-1 

df = load_data(columns_names) 

df = df.replace(np.nan, 0)

#Randomizando o dataset para pegar somente uma parcela deste para agilizar o treino
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

subSetSize = 1

end = int(len(df) * subSetSize)

df = df.iloc[:end, :]

X = df.iloc[:,0:x_end].values 

scale = StandardScaler() 

y = df.iloc[:,x_end:len(columns_names)].values 
y = np.ravel(y) # Flattening the array 

# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

# X_train = scale.fit_transform(X_train)
# X_test = scale.transform(X_test)


mean_nrmse_score = make_scorer(mean_nrmse, greater_is_better=False)
maxmin_nrmse_score = make_scorer(maxmin_nrmse, greater_is_better=False)
sd_nrmse_score = make_scorer(sd_nrmse, greater_is_better=False)
iq_nrmse_score = make_scorer(iq_nrmse, greater_is_better=False)

print("Data loaded")

regr = MLPRegressor(hidden_layer_sizes=(16,), activation='logistic', random_state=1, max_iter=500, verbose=True, solver='adam')

pipeline = Pipeline([("standardscaler", scale), ("mlpregressor", regr)])

scoring_dict = {
    "mean_nrmse_score": mean_nrmse_score,
    "maxmin_nrmse_score": maxmin_nrmse_score,
    "sd_nrmse_score": sd_nrmse_score,
    "iq_nrmse_score": iq_nrmse_score,
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

# scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
# scores = cross_validate(pipeline, X, y, cv=5, scoring=scoring_dict)
scores = cross_validate(pipeline, X, y, cv=3, scoring=scoring_dict)

# print(scores.mean())
print(scores)

for key in scores:
    print(key,scores[key].mean())

# regr.fit(X_train, y_train)

# y_pred = regr.predict(X_test)

# print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
# print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# print('Root mean squared error: %.2f' % mean_squared_error(y_test, y_pred, squared = False))
