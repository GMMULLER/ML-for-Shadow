from shadow import getShadow
import psycopg2
import pandas as pd
import numpy as np
import ast

def fetchData():
    path = '../../TCC/boston-shadows/bos-dec-21/'

    #Establishing the connection
    conn = psycopg2.connect(database="tccbase_boston", user='postgres', password='admin', host='127.0.0.1', port= '5432')

    #Setting auto commit false
    conn.autocommit = True

    #Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    # Preparing SQL queries to SELECT records
    cursor.execute("SELECT latitude,longitude FROM var_radial ORDER BY id;")
    tuplas_recuperadas = cursor.rowcount
    resultado = cursor.fetchall()

    linha = 0

    for i in range(tuplas_recuperadas):
        coordenadas = resultado[i]
        
        try:
            sombra = getShadow(path, float(coordenadas[0]), float(coordenadas[1]))
            cursor.execute("UPDATE var_radial SET dec21 = %s WHERE latitude = %s AND longitude = %s", (sombra, resultado[i][0], resultado[i][1]))
        except FileNotFoundError:
            print("linha: "+str(linha)+" fora do escopo")

        print("linha: "+str(linha))
        linha += 1

    # Commit your changes in the database
    conn.commit()
    # Closing the connection
    conn.close()

fetchData()



