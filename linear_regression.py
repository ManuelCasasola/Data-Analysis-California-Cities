from sklearn.linear_model import LinearRegression;import numpy as np
import matplotlib.pyplot as plt;import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def cargar_datos(archivo):
    # Cargamos el archivo, en este caso CSV
    df=pd.read_csv(archivo)
    # Eliminamos las filas con valores faltantes
    df=df.dropna(subset=['area_total_km2'])

    return df

def ajuste_modelo(df):
    # Seleccionamos las variables que necesitamos
    X=df['population_total'].values.reshape(-1,1)
    y=df['area_total_km2'].values
    # Creamos y ajustamos el modelo de regresión lineal
    modelo=LinearRegression()
    modelo.fit(X,y)

    return modelo

def evaluar_modelo(modelo,df):
    # Seleccionamos las variables que necesitamos
    X=df['population_total'].values.reshape(-1,1)
    y=df['area_total_km2'].values
    # Predecimos valores utilizando el modelo ajustado
    y_pred=modelo.predict(X)
    mse=mean_squared_error(y,y_pred)
    r2=r2_score(y,y_pred)

    return mse,r2

def plot_modelo(modelo,df):
    # Seleccionamos las variables que necesitamos
    X = df['population_total'].values.reshape(-1, 1)
    y = df['area_total_km2'].values
    # Predecimos valores utilizando el modelo ajustado
    y_pred = modelo.predict(X)
    # Creamos un gráfico de dispersión con los datos originales
    fig, ax = plt.subplots()
    ax.scatter(X, y, picker=5)  # Aumentamos la tolerancia de selección a 5 puntos

    # Dibujamos la linea de regresión
    ax.plot(X, y_pred, color='red')
    # Etiquetamos los ejes y título del gráfico
    ax.set_xlabel('Población total')
    ax.set_ylabel('Área total (km^2)')
    ax.set_title('Regresión Lineal')
        # Función para manejar eventos de clic en el gráfico
    def click(evento):
        # Obtenemos el índice del punto seleccionado
        ind = evento.ind[0]
        # Obtenemos la fila correspondiente del DataFrame
        row = df.iloc[ind]
        # Agregamos una etiqueta al punto con el nombre de la ciudad
        ax.annotate(row['city'], (row['population_total'], row['area_total_km2']))
        # Actualizamos el gráfico
        fig.canvas.draw()
        # Conectamos el evento 'pick_event' con la función 'click'
    fig.canvas.mpl_connect('pick_event', click)
        # Mostramos el gráfico
    plt.show()


df=cargar_datos('california_cities.csv')
modelo=ajuste_modelo(df)
evaluacion=evaluar_modelo(modelo,df)
print(evaluacion)
visualizacion=plot_modelo(modelo,df)