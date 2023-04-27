from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np;import pandas as pd
import matplotlib.pyplot as plt

def cargar_datos(archivo):
    # Accedemos al archivo CSV
    df=pd.read_csv(archivo)
    # Eliminamos las filas con valores faltantes
    df=df.dropna(subset=['area_total_km2'])

    return df

def ajuste_modelo_polynomial(df,degree=2):
    # Seleccionamos las variables necesarias
    X=df['population_total'].values.reshape(-1,1)
    y=df['area_total_km2'].values
    # Creamos características polinómicas
    poly=PolynomialFeatures(degree=degree)
    X_poly=poly.fit_transform(X)
    # Creamos y ajustamos el modelo de regresión lineal
    modelo=LinearRegression()
    modelo.fit(X_poly,y)

    return modelo,poly

def evaluar_modelo_polynomial(modelo,poly,df):
    # Seleccionamos las variables necesarias
    X=df['population_total'].values.reshape(-1,1)
    y=df['area_total_km2'].values
    # Transformamos las características utilizando el objeto PolynomialFeatures
    X_poly=poly.transform(X)
    # Predecimos valores utilizando el modelo ajustado
    y_pred=modelo.predict(X_poly)
    # Calculamos el error cuadrático medio y el coeficiente de determinación
    mse=mean_squared_error(y,y_pred)
    r2=r2_score(y,y_pred)

    return mse, r2

def plot_modelo_polynomial(modelo, poly, df):
    # Seleccionamos las variables necesarias
    X = df['population_total'].values.reshape(-1, 1)
    y = df['area_total_km2'].values
    # Transformamos las características utilizando el objeto PolynomialFeatures
    X_poly = poly.transform(X)
    # Predecimos valores utilizando el modelo ajustado
    y_pred = modelo.predict(X_poly)
    # Creamos un gráfico de dispersión con los datos originales
    fig, ax = plt.subplots()
    ax.scatter(X, y, picker=5)
    # Dibujamos la línea de regresión
    ax.plot(X, y_pred, color='red')
    # Etiquetas de los ejes y título del gráfico
    ax.set_xlabel('Población total')
    ax.set_ylabel('Área total (km^2)')
    ax.set_title('Regresión polinómica')

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

# Cargar los datos del archivo CSV
df = cargar_datos('california_cities.csv')

# Ajustar el modelo de regresión polinómica
modelo, poly = ajuste_modelo_polynomial(df)

# Visualizar la línea de regresión
plot_modelo_polynomial(modelo, poly, df)

# Error cuadrático medio y coeficiente de determinación
mse=evaluar_modelo_polynomial(modelo,poly,df)
print(mse)