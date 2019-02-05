import pandas as pd  
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def imprime_matriz(matriz):
    linhas = len(matriz)
    colunas = len(matriz[0])

    for i in range(linhas):
        for j in range(colunas):
            if(j == colunas - 1):
                print("%d" %matriz[i][j], end = "  ")
            else:
                print("%d" %matriz[i][j], end = "  ")
        print()

dataset = pd.read_csv('leaf/leaf_mod.csv')
dataset.head()

X = dataset.iloc[:, 2:16].values  
y = dataset.iloc[:, 0:1].values 

## Distribuição aleatória

n_arvo = 100
classif = RandomForestClassifier(n_estimators = n_arvo) #aqui foi-se alterado o valor do n_estimators para mudar o número de árvores
scoring = ['precision_macro', 'recall_macro']
scores = cross_validate(classif, X, y, cv=10, scoring=scoring)
print("Acurácia média:",scores['test_precision_macro'].mean())
print('\n\n\n')


## Distribuição seguencial

dataAux = np.genfromtxt('leaf/leaf_mod.csv', delimiter=',')

dataset2 = np.array(dataAux)
size = len(dataset)//10
start = 1
rand_st = 10
conf_mat = np.zeros((30, 30), dtype=np.int)
for i in range(10):
    if i == 0:
        test = dataset2[start:size]
        train = dataset2[34:]
        y_train = train[:,:1].astype(int).ravel()
        X_train = train[:,2:]
        y_test = test[:,:1].astype(int).ravel()
        X_test = test[:,2:]
        u_clf = RandomForestClassifier(random_state=rand_st)
        u_clf.fit(X_train, y_train)
        y_pred = u_clf.predict(X_test)
        print("Partição %d [%d:%d]: " %((i+1), start, size), end = ' ')
        print(accuracy_score(y_test, y_pred))
        conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        start = size
    else:
        test = dataset2[start:size+start]
        train = np.concatenate((dataset2[1:start],dataset2[size+start:]), axis=0)
        y_train = train[:,:1].astype(int).ravel()
        X_train = train[:,2:]
        y_test = test[:,:1].astype(int).ravel()
        X_test = test[:,2:]
        u_clf = RandomForestClassifier(random_state=rand_st)
        u_clf.fit(X_train, y_train)
        y_pred = u_clf.predict(X_test)
        print("Partição %d [%d:%d]: " %((i+1), start, (size+start)), end = ' ')
        print(accuracy_score(y_test, y_pred))
        conf_mat += confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
        start += size


print('\n\n\nMatriz de confusão:')
imprime_matriz(conf_mat)