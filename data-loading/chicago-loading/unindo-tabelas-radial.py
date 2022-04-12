import psycopg2

conn = psycopg2.connect(database="tccbase_2", user='postgres', password='admin', host='127.0.0.1', port= '5432')

#Setting auto commit false
conn.autocommit = True

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

cursor.execute("SELECT id FROM var_radial_chicago;")

tuplas_recuperadas = cursor.rowcount
resultado = cursor.fetchall()

line = 100

radial_table = "processed_radial_"+str(line)+"_chicago"

for i in range(tuplas_recuperadas):
    id = resultado[i][0]

    update_string = ""
    
    print('Processando id: '+str(i))

    # Para cada id passar pelas 24 tabelas line recuperando os valores calculados para aquela linha
    for index_tabela in range(1,25):

        cursor.execute('SELECT heightroof_count_'+str(index_tabela)+', heightroof_max_'+str(index_tabela)+', heightroof_mean_'+str(index_tabela)+' FROM '+radial_table+' WHERE id = '+str(id))

        campos_linha = cursor.fetchall()

        heightroof_count = ""
        heightroof_max = ""
        heightroof_mean = ""

        if campos_linha[0][0] == None:
            heightroof_count = "NULL"
        else:
            heightroof_count = campos_linha[0][0]

        if campos_linha[0][1] == None:
            heightroof_max = "NULL"
        else:
            heightroof_max = campos_linha[0][1]

        if campos_linha[0][2] == None:
            heightroof_mean = "NULL"
        else:
            heightroof_mean = campos_linha[0][2]

        update_string += 'heightroof_count_'+str(line)+'_'+str(index_tabela)+' = '+str(heightroof_count)+',\
                          heightroof_max_'+str(line)+'_'+str(index_tabela)+' = '+str(heightroof_max)+',\
                          heightroof_mean_'+str(line)+'_'+str(index_tabela)+' = '+str(heightroof_mean)

        if index_tabela != 24:
            update_string += ','

    sql = 'UPDATE var_radial_chicago SET '+update_string+' WHERE id = '+str(id)
    
    cursor.execute(sql)

    conn.commit()

conn.close()
