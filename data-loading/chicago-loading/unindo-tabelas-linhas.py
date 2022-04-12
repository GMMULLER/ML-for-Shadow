import psycopg2

conn = psycopg2.connect(database="tccbase_2", user='postgres', password='admin', host='127.0.0.1', port= '5432')

#Setting auto commit false
conn.autocommit = True

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

columns = ['heightroof_count', 'heightroof_max', 'heightroof_mean']

cursor.execute("SELECT id, line1, line2, line3, line4, line5, line6, line7, line8, line9, \
line10, line11, line12, line13, line14, line15, line16, line17, line18, line19, \
line20, line21, line22, line23, line24 FROM radial_100_chicago;")

linha = 0

tuplas_recuperadas = cursor.rowcount
resultado = cursor.fetchall()

for i in range(tuplas_recuperadas):
    print("Linha "+str(linha))

    id = resultado[i][0]
    para_inserir_1 = ()
    para_inserir_1 += resultado[i]
    # Para cada id passar pelas 24 tabelas line recuperando os valores calculados para aquela linha
    for index_tabela in range(1,25):

        nome_tabela = 'line'+str(index_tabela)

        atributos_select = ''

        for index_column,column in enumerate(columns):
            if index_column != len(columns)-1:
                atributos_select += column+','
            else:
                atributos_select += column

        cursor.execute("SELECT "+atributos_select+" FROM "+nome_tabela+" WHERE id = "+str(id)+";")

        para_inserir_1 += cursor.fetchall()[0]

    # inserir para_inserir_1 na tabela
    values_insert_1 = '('
    for j in range(len(para_inserir_1)):
        if j != len(para_inserir_1)-1:
            values_insert_1 += '%s,'
        else:
            values_insert_1 += '%s'
    values_insert_1 += ')'

    cursor.execute("INSERT INTO processed_radial_100_chicago VALUES "+values_insert_1,para_inserir_1)

    conn.commit()

    linha += 1

conn.close()
