import matplotlib.pyplot as plt
import psycopg2
import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Establishing the connection
conn = psycopg2.connect(database="tccbase", user='postgres', password='admin', host='127.0.0.1', port= '5432')

#Setting auto commit false
conn.autocommit = True

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

sql = "SELECT exposure,june21 FROM complete_sky_exposure WHERE exposure IS NOT NULL AND june21 IS NOT NULL;"

cursor.execute(sql)

colnames = [desc[0] for desc in cursor.description]
tuplas_recuperadas = cursor.rowcount
resultado = cursor.fetchall()

col1 = []
col2 = []

for linha in resultado:
    col1.append(float(linha[0]))
    col2.append(float(linha[1]))

d = {colnames[0]:col1, colnames[1]:col2}
df = pd.DataFrame(data=d)

X = df.iloc[:,0:1].values
Y = df.iloc[:,1:2].values

# fazendo com que os valores variem de 0-1 para usar sqrt X
scale = MinMaxScaler()
X = scale.fit_transform(X)

# 20% dos dados vao ser usados para teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)

poly = PolynomialFeatures(degree=1)
# feature X modificado
x_poly = poly.fit_transform(X_train)

# sqr_root_x_train = []

# for element in X_train:
#     sqr_root_x_train.append([math.sqrt(element[0])])
#     # sqr_root_x_train.append([pow(element[0],0.7)])

# x_poly = np.append(x_poly,sqr_root_x_train,1)

poly.fit(X_train, Y_train) # nao sei o que isso faz

model = LinearRegression()
model.fit(x_poly, Y_train)

x_test_poly = poly.fit_transform(X_test)

# sqr_root_x_test = []

# for element in X_test:
#     sqr_root_x_test.append([math.sqrt(element[0])])
#     # sqr_root_x_test.append([pow(element[0],0.82)])

# x_test_poly = np.append(x_test_poly,sqr_root_x_test,1)

y_pred = model.predict(x_test_poly)

# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print('Root Mean Square Error: %.2f' % mean_squared_error(Y_test, y_pred, squared=False)) #RMSE 
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(Y_test, y_pred)) #R2
print('Mean Absolute Error: %.2f' % mean_absolute_error(Y_test, y_pred)) #MAE

X_test = X_test.reshape(1,len(X_test))[0]
y_pred = y_pred.reshape(1,len(y_pred))[0]
Y_test = Y_test.reshape(1,len(Y_test))[0]

plt.scatter(X_test, Y_test, color='black')

# ordenando X_test e y_pred em conjunto tendo como base X_test
X_test, y_pred = (list(t) for t in zip(*sorted(zip(X_test, y_pred))))

plt.plot(X_test, y_pred, color='blue')
plt.show()

# se houver muitas input features considerar usar regularization
