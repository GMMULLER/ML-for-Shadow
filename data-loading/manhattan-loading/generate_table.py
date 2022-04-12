columns = ['heightroof_count', 'heightroof_max', 'heightroof_mean']
types = ['integer', 'numeric', 'numeric']    

file = open('colunas_tabela.txt','w')

for j in [10,40,70,100]:
    for i in range(1,25):
        for index, column in enumerate(columns):
            file.write(column+'_'+str(j)+'_'+str(i)+' '+types[index]+','+'\n')

file.close()