import pandas as pd;from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans;import matplotlib.pyplot as plt

def cluster_ciudades(archivo,columnas,n_clusters=3):
    # Cargamos el archivo CSV en un DataFrame
    df=pd.read_csv(archivo)
    # Seleccionamos las columnas que necesitamos
    datos=df[columnas]
    datos=datos.dropna()
    # Estandarizamos los datos
    scaler=StandardScaler()
    datos_std=scaler.fit_transform(datos)
    # Creamos el modelo de clustering K-Means
    # con n_clusters grupos
    kmeans=KMeans(n_clusters=n_clusters)
    # Ajustamos el modelo a los datos estandarizados
    kmeans.fit(datos_std)
    # Predecimos los grupos de cada ciudad
    grupos=kmeans.predict(datos_std)

    return grupos

def plot_clusters(df, x_col, y_col, grupos):
    # Creamos un gráfico de dispersión con los datos especificados
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col], c=grupos, picker=5)
    # Etiquetas de los ejes y título del gráfico
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title('Clustering de ciudades por ubicación geográfica')

    # Función para manejar eventos de clic en el gráfico
    def click(evento):
        # Obtenemos el índice del punto seleccionado
        ind = evento.ind[0]
        # Obtenemos la fila correspondiente del DataFrame
        row = df.iloc[ind]
        # Agregamos una etiqueta al punto con el nombre de la ciudad
        ax.annotate(row['city'], (row[x_col], row[y_col]))
        # Actualizamos el gráfico
        fig.canvas.draw()

    # Conectamos el evento 'pick_event' con la función 'click'
    fig.canvas.mpl_connect('pick_event', click)

    # Mostramos el gráfico
    plt.show()

 # Ejemplo de uso de las funciones
df = pd.read_csv('california_cities.csv')
df = df.dropna(subset=['latd', 'longd', 'elevation_m'])
groups = cluster_ciudades('california_cities.csv', ['latd', 'longd', 'elevation_m'], n_clusters=3)
plot_clusters(df, 'longd', 'latd', groups)   


