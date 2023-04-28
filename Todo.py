import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


#CARGAR DATOS CSV
datos4=pd.read_csv('r4.csv',sep=";",decimal=",")
datos3=pd.read_csv('r3.csv',sep=";",decimal=",")
datos2=pd.read_csv('r2.csv',sep=";",decimal=",")
datos1=pd.read_csv('r1.csv',sep=";",decimal=",")

#CARGAMOS LOS VALORES DE LAS INTENSIDADES DE CORRIENTE DE CADA DATAFRAME
y4=datos4['I(Intensidad)']
y2=datos2['I(Intensidad)']
y3=datos3['I(Intensidad)']
y1=datos1['I(Intensidad)']

x=datos1['V(Voltaje)']

#COFIGURAMOS EL DIAGRAMA DE DISPERSIÓN
plt.ylabel("Corriente(mA)")
plt.xlabel("Potencial Electrico(V)")
plt.xticks(range(16))

#ESTABLECEMOS EL DIAGRAMA DE DISPERSIÓN
plt.scatter(x,y=y4,c="r",label="R4")
plt.scatter(x,y=y2,c="b",label="R2")
plt.scatter(x,y=y3,c="g",label="R3")
plt.scatter(x,y=y1,c="k",label="R1")


#CREAMOS LAS REGRESIONES LINEALES DE CADA MODELO
regresion1=linear_model.LinearRegression()
regresion2=linear_model.LinearRegression()
regresion3=linear_model.LinearRegression()
regresion4=linear_model.LinearRegression()


#CREAMOS LOS CUATRO MODELOS
modelo1=regresion1.fit(x.values.reshape(-1,1),y1)
modelo2=regresion2.fit(x.values.reshape(-1,1),y2)
modelo3=regresion3.fit(x.values.reshape(-1,1),y3)
modelo4=regresion4.fit(x.values.reshape(-1,1),y4)


y4_pred=regresion4.predict(x.values.reshape(-1,1))
y2_pred=regresion2.predict(x.values.reshape(-1,1))
y3_pred=regresion3.predict(x.values.reshape(-1,1))
y1_pred=regresion1.predict(x.values.reshape(-1,1))

plt.plot(x,y1_pred,color="k",linewidth=2)
plt.plot(x,y2_pred,color="b",linewidth=2)
plt.plot(x,y3_pred,color="g",linewidth=2)
plt.plot(x,y4_pred,color="r",linewidth=2)

plt.legend()
plt.show()
