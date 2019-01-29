# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:24:36 2018

@author: nicol
"""

import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from mylib import mylib
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import (accuracy_score,precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
import scipy.spatial.distance as sc
from scipy.cluster import hierarchy
from sklearn import datasets

#%% 1. Importar datos.
data = pd.read_csv('C:/Users/nicol/Documents/Tareas Iteso 5/Ciencia de datos/Proyecto final/HR_comma_sep.csv')

#%% 2. Limpiar los datos y dar el reporte de calidad.
dqr_data = mylib.dqr(data)

#%% Reemplazar texto con función
def replace_text(x,to_replace,replacement):
    try:
        x = x.replace(to_replace,replacement)
    except:
        pass
    return x 

#%%  Reemplazamos 
data = replace_text(data,'low',0)
data = replace_text(data,'medium',1)
data = replace_text(data,'high',2)
data = replace_text(data,'sales',0)
data = replace_text(data,'accounting',1)
data = replace_text(data,'hr',2)
data = replace_text(data,'technical',3)
data = replace_text(data,'support',4)
data = replace_text(data,'management',5)
data = replace_text(data,'IT',6)
data = replace_text(data,'product_mng',7)
data = replace_text(data,'marketing',8)
data = replace_text(data,'RandD',9)

#%% 3. Diseñe un modelo logístico de clasificación considerando todas las variables que se proporcionan (Modelo 1). 
X = data.iloc[:,0:6].join(data.iloc[:,7:])
Y = data.iloc[:,6]

#Normalizar
Xn = (X-X.mean())/X.std()

#%% Buscar el grado del polinomio óptimo (Lo mejor)
modelo = linear_model.LogisticRegression(C=1)
grados = np.arange(1,5)
Accu = np.zeros(grados.shape)
Prec = np.zeros(grados.shape)
Reca = np.zeros(grados.shape)
Nvar = np.zeros(grados.shape)

for ngrado in grados:
    poly = PolynomialFeatures(ngrado)
    Xa = poly.fit_transform(Xn)
    modelo.fit(Xa,Y)
    Yhat = modelo.predict(Xa)
    Accu[ngrado-1] = accuracy_score(Y,Yhat)
    Prec[ngrado-1] = precision_score(Y,Yhat)
    Reca[ngrado-1] = recall_score(Y,Yhat)
    Nvar[ngrado-1] = len(modelo.coef_[0]) #Número de parámetros o w's del modelo.
    
plt.plot(grados,Accu)
plt.plot(grados,Prec)
plt.plot(grados,Reca)
plt.xlabel('Grado Polinomio')
plt.ylabel('% aciertos')
plt.legend(('Accuracy', 'Precision', 'Recall'), loc='best')
plt.grid()
plt.show()

#%% Selecionar el modelo deseado
ngrado = 2
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(Xn)
modelo = linear_model.LogisticRegression(C=1e20)
modelo.fit(Xa,Y)

W = modelo.coef_[0]
plt.bar(np.arange(len(W)),np.abs(W))
plt.title('Datos primera regresión')
plt.grid()
plt.show()

#%% Optimización versión 1.Seleccionar coeficientes 
umbral = 0.5
ind = np.abs(W)>umbral
Xa_simplificada = Xa[:,ind]
ngrado = 2
modelo1 = linear_model.LogisticRegression(C=1e20)

modelo1.fit(Xa_simplificada,Y)
Yhat1 = modelo1.predict(Xa_simplificada)

#%% Medidas modelo optimizado 1
print(accuracy_score(Y,Yhat1))
print(recall_score(Y,Yhat1))
print(precision_score(Y,Yhat1))

#%% Aplicar técnicas de selección de las variables para decidir las variables más relevantes. (Decidimos aplicar PCA)
#Aplicar PCA
data2 = data.iloc[:,3]
data2_norm = (data2-data2.mean())/data2.std()
data.iloc[:,3] = data2_norm
m_cov = np.cov(data.transpose())
w,v = np.linalg.eig(m_cov)

#%% Proyectar los digitos en 1D
componentes = w[0:3] #en cuanto lo comprimo
m_trans = v[:,0:3] # Convertir valores a vectores propios.
data_new = np.array(np.matrix(data)*np.matrix(m_trans))

#%% Segunda Regresión logística.
modelo = linear_model.LogisticRegression(C=1)
grados = np.arange(1,5)
Accu = np.zeros(grados.shape)
Prec = np.zeros(grados.shape)
Reca = np.zeros(grados.shape)
Nvar = np.zeros(grados.shape)

for ngrado in grados:
    poly = PolynomialFeatures(ngrado)
    Xa = poly.fit_transform(data_new)
    modelo.fit(Xa,Y)
    Yhat = modelo.predict(Xa)
    Accu[ngrado-1] = accuracy_score(Y,Yhat)
    Prec[ngrado-1] = precision_score(Y,Yhat)
    Reca[ngrado-1] = recall_score(Y,Yhat)
    Nvar[ngrado-1] = len(modelo.coef_[0]) #Número de parámetros o w's del modelo.
    
plt.plot(grados,Accu)
plt.plot(grados,Prec)
plt.plot(grados,Reca)
plt.xlabel('Grado Polinomio')
plt.ylabel('% aciertos')
plt.legend(('Accuracy', 'Precision', 'Recall'), loc='best')
plt.grid()
plt.show()

#%% Selecionar el modelo deseado
ngrado = 2
poly = PolynomialFeatures(ngrado)
Xa = poly.fit_transform(data_new)
modelo = linear_model.LogisticRegression(C=1e20)
modelo.fit(Xa,Y)

W = modelo.coef_[0]
plt.bar(np.arange(len(W)),np.abs(W))
plt.title('Datos segunda regresión')
plt.grid()
plt.show()

#%% Optimización versión 1.Seleccionar coeficientes 
umbral = 0.5
ind = np.abs(W)>umbral
Xa_simplificada = Xa[:,ind]
ngrado = 2
modelo1 = linear_model.LogisticRegression(C=1e20)

modelo1.fit(Xa_simplificada,Y)
Yhat1 = modelo1.predict(Xa_simplificada)


#%% Medidas modelo optimizado 2
print(accuracy_score(Y,Yhat1))
print(recall_score(Y,Yhat1))
print(precision_score(Y,Yhat1))

#%% 3. Tercer modelo (Optimización) 
X3 = Xn
Y = Y = data.iloc[:,6] #Volvemos a tomar los datos para optimizar. 

#%% Seleccionar entrenamiento y prueba
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X3,Y,test_size=0.4,random_state=0)

#%% Regresión logística 
modelo_rl3 = linear_model.LogisticRegression()
ngrado = 2
poly3 = PolynomialFeatures(ngrado)
Xa3 = poly3.fit_transform(Xtrain) 

#%% Sacar scores de mi modelo.
modelo_rl3.fit(Xa3,Ytrain)
Yhat_train = modelo_rl3.predict(Xa3)

print(accuracy_score(Ytrain,Yhat_train))
print(precision_score(Ytrain,Yhat_train))
print(recall_score(Ytrain,Yhat_train))

#%% Aplicar técnicas de selección de las variables para decidir las variables más relevantes. (Decidimos aplicar PCA)
#Aplicar PCA
data4 = data.iloc[:,3]
data4_norm = (data4-data4.mean())/data4.std()
data.iloc[:,3] = data4_norm
m_cov = np.cov(data.transpose())
w,v = np.linalg.eig(m_cov)

#%% Proyectar los digitos en 1D
componentes4 = w[0:3] #en cuanto lo comprimo
m_trans4 = v[:,0:3] # Convertir valores a vectores propios.
data_new4 = np.array(np.matrix(data)*np.matrix(m_trans))

#%% 4. Cuarto modelo (Optimización) 
X4 = data_new4
Y = Y = data.iloc[:,6] #Volvemos a tomar los datos para optimizar. 

#%% Seleccionar entrenamiento y prueba
Xtrain4,Xtest4,Ytrain4,Ytest4 = train_test_split(X4,Y,test_size=0.4,random_state=0)

#%% Regresión logística 
modelo_rl4 = linear_model.LogisticRegression()
ngrado = 2
poly4 = PolynomialFeatures(ngrado)
Xa4 = poly4.fit_transform(Xtrain4) 

#%% Sacar scores de mi modelo.
modelo_rl4.fit(Xa4,Ytrain4)
Yhat_train4 = modelo_rl4.predict(Xa4)

print(accuracy_score(Ytrain4,Yhat_train4))
print(precision_score(Ytrain4,Yhat_train4))
print(recall_score(Ytrain4,Yhat_train4))

#%% 5. Quinto y último modelo. 
X5 = Xn
Y = data.iloc[:,6] #Volvemos a tomar los datos para optimizar. 
Xtrain5,Xtest5,Ytrain5,Ytest5 = train_test_split(X5,Y,test_size=0.35,random_state=0)

#%% Crear y entrenar el modelo SVM
modelo = svm.SVC(kernel='rbf')
modelo.fit(X,Y)

Yhat_svm = modelo.predict(X)
print(accuracy_score(Y,Yhat_svm))
print(recall_score(Y,Yhat_svm))
print(precision_score(Y,Yhat_svm))

