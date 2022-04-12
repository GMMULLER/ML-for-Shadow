import matplotlib.pyplot as plt
import psycopg2
import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

scale = StandardScaler()
X = scale.fit_transform(X)

print(X)

# 20% dos dados vao ser usados para teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)

poly = PolynomialFeatures(degree=3)
# feature X modificado
x_poly = poly.fit_transform(X_train)

poly.fit(X_train, Y_train) # nao sei o que isso faz

model = LinearRegression()
model.fit(x_poly, Y_train)

y_pred = model.predict(poly.fit_transform(X_test))

# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(Y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f' % r2_score(Y_test, y_pred))

X_test = X_test.reshape(1,len(X_test))[0]
y_pred = y_pred.reshape(1,len(y_pred))[0]
Y_test = Y_test.reshape(1,len(Y_test))[0]

plt.scatter(X_test, Y_test, color='black')

# ordenando X_test e y_pred em conjunto tendo como base X_test
X_test, y_pred = (list(t) for t in zip(*sorted(zip(X_test, y_pred))))

plt.plot(X_test, y_pred, color='blue')
plt.show()

