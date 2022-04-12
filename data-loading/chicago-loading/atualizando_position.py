import psycopg2

conn = psycopg2.connect(database="tccbase_2", user='postgres', password='admin', host='127.0.0.1', port= '5432')

#Setting auto commit false
conn.autocommit = True

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

cursor.execute("SELECT id FROM var_radial_chicago;")

tuplas_recuperadas = cursor.rowcount
resultado = cursor.fetchall()

for i in range(tuplas_recuperadas):
    id = resultado[i][0]

    print('Processando id: '+str(i))

    update_string = ''

    cursor.execute('SELECT centroid FROM sub_100_centroid_segments_chicago WHERE id = '+str(id))

    campos_linha = cursor.fetchall()

    update_string = "position = '"+str(campos_linha[0][0])+"'"

    sql = 'UPDATE var_radial_chicago SET '+update_string+' WHERE id = '+str(id)
    
    cursor.execute(sql)

    conn.commit()

conn.close()
