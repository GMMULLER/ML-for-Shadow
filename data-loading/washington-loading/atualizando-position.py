import psycopg2

conn = psycopg2.connect(database="tccbase_dc", user='postgres', password='admin', host='127.0.0.1', port= '5432')

#Setting auto commit false
conn.autocommit = True

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

cursor.execute("SELECT id FROM var_radial;")

tuplas_recuperadas = cursor.rowcount
resultado = cursor.fetchall()

for i in range(tuplas_recuperadas):
    id = resultado[i][0]

    print('Processando id: '+str(i))

    update_string = ''

    cursor.execute('SELECT position FROM streets WHERE id = '+str(id))

    campos_linha = cursor.fetchall()

    update_string = "position = '"+str(campos_linha[0][0])+"'"

    sql = 'UPDATE var_radial SET '+update_string+' WHERE id = '+str(id)
    
    cursor.execute(sql)

    conn.commit()

conn.close()
