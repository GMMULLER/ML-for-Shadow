from os import pipe
import psycopg2
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_validate

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

    # sql = "SELECT "+select_columns+" FROM complete_radial_buffer_sky_exposure_1 as c1, complete_radial_buffer_sky_exposure_2 as c2 WHERE "+where_clause+" AND c1.id = c2.id;"
    sql = "SELECT "+select_columns+" FROM complete_radial_buffer_sky_exposure_1 as c1, complete_radial_buffer_sky_exposure_2 as c2 WHERE c1.id = c2.id;"

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

# columns_names = ['heightroof_max', 'heightroof_sum', 
#                 'heightroof_mean', 'groundelev_max', 
#                 'groundelev_sum', 'groundelev_mean',  
#                 'shape_area_max', 'shape_area_sum', 'shape_area_mean',
#                 'shape_len_max', 'shape_len_sum', 'shape_len_mean',
#                 'exposure', 'june21']

# columns = ['heightroof_count', 'heightroof_min', 'heightroof_max', 'heightroof_sum', 'heightroof_mean', 'heightroof_median', 'heightroof_stddev', 'heightroof_q1', 'heightroof_q3', 'heightroof_iqr', 'groundelev_count', 'groundelev_min',
# 'groundelev_max', 'groundelev_sum', 'groundelev_mean', 'groundelev_median', 'groundelev_stddev', 'groundelev_q1', 'groundelev_q3', 'groundelev_iqr', 'shape_area_count', 'shape_area_min', 'shape_area_max', 'shape_area_sum', 'shape_area_mean',
# 'shape_area_median', 'shape_area_stddev', 'shape_area_q1', 'shape_area_q3', 'shape_area_iqr', 'shape_len_count', 'shape_len_min', 'shape_len_max', 'shape_len_sum', 'shape_len_mean', 'shape_len_median', 'shape_len_stddev', 'shape_len_q1',
# 'shape_len_q3', 'shape_len_iqr']

# columns = ['heightroof_count', 'heightroof_min', 'heightroof_max', 'heightroof_sum', 'heightroof_mean', 'heightroof_median', 'heightroof_stddev',
# 'shape_area_count', 'shape_area_min', 'shape_area_max', 'shape_area_sum', 'shape_area_mean',
# 'shape_area_median', 'shape_area_stddev', 'shape_len_count', 'shape_len_min', 'shape_len_max', 'shape_len_sum', 'shape_len_mean', 'shape_len_median', 'shape_len_stddev']

columns = [
    'heightroof_count', 'heightroof_max', 'heightroof_mean'
]

columns_names = ['exposure']

for i in range(1,26):
    if i != 19: #linha 19 e 20 sao iguais
        addCount = True # soh pode ter uma contagem por linha
        for toAdd in columns:
            if "count" not in toAdd:
                columns_names.append(toAdd+"_"+str(i))
            elif addCount:
                columns_names.append(toAdd+"_"+str(i))
                addCount = False

columns_names.append("june21")

x_end = len(columns_names)-1 

df = load_data(columns_names) 

df = df.replace(np.nan, 0)

X = df.iloc[:,0:x_end].values 

scale = StandardScaler() 

y = df.iloc[:,x_end:len(columns_names)].values 
y = np.ravel(y) # Flattening the array 

# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

# X_train = scale.fit_transform(X_train)
# X_test = scale.transform(X_test)

regr = MLPRegressor(hidden_layer_sizes=(16), activation='logistic', random_state=1, max_iter=500, verbose=True)

pipeline = Pipeline([("standardasacler", scale), ("mlpregressor", regr)])

# scores = cross_val_score(pipeline, X, y, scoring="neg_root_mean_squared_error")

scores = cross_validate(pipeline, X, y, scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'] )

print(scores)

# regr.fit(X_train, y_train)

# y_pred = regr.predict(X_test)

# print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
# print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# print('Root mean squared error: %.2f' % mean_squared_error(y_test, y_pred, squared = False))
