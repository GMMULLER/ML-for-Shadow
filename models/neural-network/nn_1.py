import psycopg2
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

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

columns_names = ['heightroof_max', 'heightroof_sum', 
                'heightroof_mean', 'groundelev_max', 
                'groundelev_sum', 'groundelev_mean',  
                'shape_area_max', 'shape_area_sum', 'shape_area_mean',
                'shape_len_max', 'shape_len_sum', 'shape_len_mean',
                'exposure', 'june21']

# columns_names = ['heightroof_sum', 'groundelev_sum', 'exposure', 'june21']

x_end = len(columns_names)-1

df = load_data(columns_names)

X = df.iloc[:,0:x_end].values

# fazendo com que os valores variem de [0-1]
scale = StandardScaler()
X = scale.fit_transform(X)

y = df.iloc[:,x_end:len(columns_names)].values
y = np.ravel(y) # Flattening the array

# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

# param_list = {
#     "hidden_layer_sizes": [(10,),(15,),(20,)],
#     "activation": ["identity", "logistic", "tanh", "relu"],
#     "solver": ["lbfgs", "sgd", "adam"],
#     "learning_rate": ["constant", "invscaling", "adaptive"],
#     "max_iter": [500],
#     "random_state": [1]
# }

param_list = {
    "hidden_layer_sizes": [(16,)],
    "activation": ["logistic"],
    "solver": ["adam"],
    "max_iter": [500],
    "random_state": [1],
    "verbose": [True]
}

regr = MLPRegressor()

clf = GridSearchCV(regr, param_list, scoring="neg_mean_absolute_error")

# ["neg_root_mean_squared_error","r2","neg_mean_absolute_error"]
clf.fit(X, y)

print(clf.best_estimator_)
print("==================")
print(clf.cv_results_)
print("==================")
print(clf.best_score_)
print("==================")
print(clf.best_params_)

# y_pred = regr.predict(X_test)

# print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

# cross-validation
# y = np.ravel(y) # Flattening the array

# # reescalling X values to [0,1]
# scale = MinMaxScaler()

# regr = MLPRegressor(hidden_layer_sizes=16, random_state=1, max_iter=500)

# pipeline = Pipeline([("minmaxscaler", scale), ("neural-network-regressors", regr)])

# scores = cross_val_score(pipeline, X, y, scoring="r2", cv=10)

# print("R2: "+str(scores.mean()))

# =============== Tentativas ====================
# train_test_split (random_state = 1)
# ['heightroof_sum', 'groundelev_sum', 'exposure', 'june21']
# MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(16,))
# Não teve problema quanto a não convergir
# r2_score = 0.55
# -----------------------------------------------
# train_test_split (random_state = 1)
# ['heightroof_min', 'heightroof_max', 'heightroof_sum', 
# 'heightroof_mean', 'groundelev_min', 'groundelev_max', 
# 'groundelev_sum', 'groundelev_mean', 'shape_area_min', 
# 'shape_area_max', 'shape_area_sum', 'shape_area_mean',
# 'shape_len_min', 'shape_len_max', 'shape_len_sum', 'shape_len_mean',
# 'exposure', 'june21']   
# MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(16,))
# Não convergiu
# r2_score = -0.02
# -----------------------------------------------
# train_test_split (random_state = 1)
# ['heightroof_min', 'heightroof_max', 'heightroof_sum', 
# 'heightroof_mean', 'groundelev_min', 'groundelev_max', 
# 'groundelev_sum', 'groundelev_mean', 'shape_area_min', 
# 'shape_area_max', 'shape_area_sum', 'shape_area_mean',
# 'shape_len_min', 'shape_len_max', 'shape_len_sum', 'shape_len_mean',
# 'exposure', 'june21']   
# MLPRegressor(random_state=1, activation="logistic", max_iter=500, hidden_layer_sizes=(16,))
# Activation function para a hidden layer sendo logistic ajudou nesse caso
# r2_score = 0.28
# -----------------------------------------------
# train_test_split (random_state = 1)
# ['heightroof_sum', 'groundelev_sum', 'exposure', 'june21']
# MLPRegressor(random_state=1, activation="logistic", max_iter=500, hidden_layer_sizes=(16,))
# r2_score = 0.55
# -----------------------------------------------
# train_test_split (random_state = 1)
# ['heightroof_sum', 'groundelev_sum', 'exposure', 'june21']
# MinMaxScaler
# MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(16,))
# r2_score = 0.56
# Ele alcanca o maximo de iteracoes (500) sem convergir segundo o criterio do scikitlearn
# -----------------------------------------------
# train_test_split (random_state = 1)
# ['heightroof_sum', 'groundelev_sum', 'exposure', 'june21']
# MinMaxScaler
# MLPRegressor(random_state=1,  activation="logistic", max_iter=500, hidden_layer_sizes=(16,))
# r2_score = 0.55
# Ele alcanca o maximo de iteracoes (500) sem convergir segundo o criterio do scikitlearn
# -----------------------------------------------
# train_test_split (random_state = 1)
# ['heightroof_min', 'heightroof_max', 'heightroof_sum', 
# 'heightroof_mean', 'groundelev_min', 'groundelev_max', 
# 'groundelev_sum', 'groundelev_mean', 'shape_area_min', 
# 'shape_area_max', 'shape_area_sum', 'shape_area_mean',
# 'shape_len_min', 'shape_len_max', 'shape_len_sum', 'shape_len_mean',
# 'exposure', 'june21']  
# MinMaxScaler
# MLPRegressor(random_state=1, activation = 'logistic', max_iter=500, hidden_layer_sizes=(16,), verbose=True).fit(X_train, y_train)
# r2_score = 0.59
# Ele alcanca o maximo de iteracoes (500) sem convergir segundo o criterio do scikitlearn
# -----------------------------------------------
# train_test_split (random_state = 1)
# ['heightroof_min', 'heightroof_max', 'heightroof_sum', 
# 'heightroof_mean', 'groundelev_min', 'groundelev_max', 
# 'groundelev_sum', 'groundelev_mean', 'shape_area_min', 
# 'shape_area_max', 'shape_area_sum', 'shape_area_mean',
# 'shape_len_min', 'shape_len_max', 'shape_len_sum', 'shape_len_mean',
# 'exposure', 'june21']  
# MinMaxScaler
# MLPRegressor(random_state=1, activation = 'logistic', max_iter=1000, hidden_layer_sizes=(16,), verbose=True).fit(X_train, y_train)
# r2_score = 0.59
# Ele alcanca o maximo de iteracoes (1000) sem convergir segundo o criterio do scikitlearn
# -----------------------------------------------
# train_test_split (random_state = 1)
# ['heightroof_max', 'heightroof_sum', 
# 'heightroof_mean', 'groundelev_max', 
# 'groundelev_sum', 'groundelev_mean',  
# 'shape_area_max', 'shape_area_sum', 'shape_area_mean',
# 'shape_len_max', 'shape_len_sum', 'shape_len_mean',
# 'exposure', 'june21']
# param_list = {
#     "hidden_layer_sizes": [(16,)],
#     "activation": ["logistic"],
#     "solver": ["adam"],
#     "max_iter": [500],
#     "random_state": [1],
#     "verbose": [True]
# }
# MinMaxScaler
# r2_score = 0.5817667244795474
# Ele alcanca o maximo de iteracoes (500) sem convergir segundo o criterio do scikitlearn
# ----------------------------------------
# train_test_split (random_state = 1)
# ['heightroof_max', 'heightroof_sum', 
# 'heightroof_mean', 'groundelev_max', 
# 'groundelev_sum', 'groundelev_mean',  
# 'shape_area_max', 'shape_area_sum', 'shape_area_mean',
# 'shape_len_max', 'shape_len_sum', 'shape_len_mean',
# 'exposure', 'june21']
# param_list = {
#     "hidden_layer_sizes": [(16,)],
#     "activation": ["logistic"],
#     "solver": ["adam"],
#     "max_iter": [500],
#     "random_state": [1],
#     "verbose": [True]
# }
# StandardScaler
# r2_score = 0.6
# Ele alcanca o maximo de iteracoes (500) sem convergir segundo o criterio do scikitlearn
# ---------------------------------------------
# train_test_split (random_state = 1)
# ['heightroof_max', 'heightroof_sum', 
# 'heightroof_mean', 'groundelev_max', 
# 'groundelev_sum', 'groundelev_mean',  
# 'shape_area_max', 'shape_area_sum', 'shape_area_mean',
# 'shape_len_max', 'shape_len_sum', 'shape_len_mean',
# 'exposure', 'june21']
# param_list = {
#     "hidden_layer_sizes": [(16,)],
#     "activation": ["logistic"],
#     "solver": ["adam"],
#     "max_iter": [500],
#     "random_state": [1],
#     "verbose": [True]
# }
# RobustScaler
# r2_score = 0.6
# Convergiu