import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def cargar_datos(archivo):
    # Cargamos los datos, en este caso un archivo CSV
    datos=pd.read_csv(archivo)
    return datos

def preprocesamiento(datos):
    # Preprocesamos los datos para eliminar valores atípicos y valores faltantes
    
    # Eliminamos valores atípicos
    datos=datos[np.abs(datos['area_water_percent'] - datos['area_water_percent'].mean()) <= (3 * datos['area_water_percent'].std())]
    # Imputamos valores faltantes
    datos.fillna(datos.mean(),inplace=True)
    return datos

def dividir_datos(datos,vars,target,test_size=0.2):
    # Dividimos el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train,X_test,y_train,y_test=train_test_split(datos[vars],datos[target],test_size=test_size)
    return X_train,X_test,y_train,y_test

def construir_modelo():
    # Construimos un pipeline con un escalador y un modelo de regresión
    pipeline=Pipeline([
        ('scaler',StandardScaler()),
        ('reg',DecisionTreeRegressor())
    ])
    # Definimos los valores de los hiperparámetros a probar
    param_grid={ 
        'reg__max_depth':[3,5,10],
        'reg__min_samples_split':[2,5,10]
    }
    # Utilizamos la validación cruzada para encontrar los mejores hiperparámetros
    grid_search=GridSearchCV(pipeline,param_grid,cv=5)
    return grid_search

def entrenar_modelo(modelo,X_train,y_train):
    # Entrenamos el modelo con los datos de entrenamiento
    modelo.fit(X_train,y_train)
    return modelo

def genera_predicciones(modelo,X_test):
    # Hacemos predicciones con el modelo entrenado en los datos de prueba
    y_pred=modelo.predict(X_test)
    return y_pred

def evaluamos_modelo(y_test,y_pred):
    # Evaluamos la precisión del modelo comparando las predicciones con los valores reales
    ecm=mean_squared_error(y_test,y_pred)
    return ecm

# Cargamos el conjunto de datos
datos=cargar_datos('california_cities.csv')
# Preprocesamos el conjunto de datos
datos=preprocesamiento(datos)
# Seleccionamos las características y la variable objetivo
vars=['latd', 'longd', 'elevation_m', 'elevation_ft', 'population_total', 'area_total_sq_mi', 'area_land_sq_mi', 'area_water_sq_mi', 'area_total_km2', 'area_land_km2', 'area_water_km2']
target='area_water_percent'
# Dividimos el conjunto de datos en conjuntos de entrenamiento y prueba
X_train,X_test,y_train,y_test=dividir_datos(datos,vars,target)
# Construimos el modelo
modelo=construir_modelo()
# Entrenamos el modelo
modelo=entrenar_modelo(modelo,X_train,y_train)
# Hacemos predicciones en el conjunto de prueba
y_pred=genera_predicciones(modelo,X_test)
# Evaluamos la precisión del modelo
ecm=evaluamos_modelo(y_test,y_pred)
print(f'Error cuadrático medio: {ecm}')

# Creamos un DataFrame con las predicciones y los valores reales
resultados=pd.DataFrame({'Predicción':y_pred,'Actual':y_test})
# Guardamos el DataFrame en un archivo CSV
resultados.to_csv('predicciones.csv',index=False)


