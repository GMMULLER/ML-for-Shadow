import psycopg2

conn = psycopg2.connect(database="tccbase", user='postgres', password='admin', host='127.0.0.1', port= '5432')

#Setting auto commit false
conn.autocommit = True

#Creating a cursor object using the cursor() method
cursor = conn.cursor()

for i in [10,40,70,100]:
    for j in range(1,25):

        line_name = "line"+str(i)+'_'+str(j)+'_idx'

        print('Gerando indice para a linha '+line_name)

        cursor.execute('CREATE INDEX IF NOT EXISTS '+line_name+' ON var_radial_buffer_sky_exposure USING GIST ('+line_name+')')


        conn.commit()

conn.close()
