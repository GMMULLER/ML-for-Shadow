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

    update_string = ""
    
    print('Processando id: '+str(i))

    update_string = ''

    # Para cada id passar pelas 24 tabelas line recuperando os valores calculados para aquela linha
    for index_tabela in [10,40,70,100]:
        radial_table = "processed_radial_"+str(index_tabela)+"_chicago"

        cursor.execute('SELECT line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12\
        , line13, line14, line15, line16, line17, line18, line19, line20, line21, line22, line23, line24 FROM '+radial_table+' WHERE id = '+str(id))

        campos_linha = cursor.fetchall()

        for index,linha in enumerate(campos_linha[0]):
            update_string += "line"+str(index_tabela)+'_'+str(index+1)+" = '"+str(linha)+"'"  

            if index_tabela != 100 or index != 23:
                update_string += ','

    sql = 'UPDATE var_radial_chicago SET '+update_string+' WHERE id = '+str(id)
    
    cursor.execute(sql)

    conn.commit()

conn.close()
