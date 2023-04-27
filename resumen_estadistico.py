import pandas as pd;import matplotlib.pyplot as plt
import mplcursors

def cargar_datos(archivo):
    # Cargamos los datos, en este caso un archivo CSV
    datos=pd.read_csv(archivo)
    return datos

df=cargar_datos('california_cities.csv')

# Media poblacional
media_poblacional=df['population_total'].mean()
print('--- Media poblacional ---')
print(media_poblacional,'habitantes de media')

# Media área ciudades
area=df['area_total_sq_mi'].mean()
print('--- Media área ciudades ---')
print(area,'mi cuadrados')

# Correlación población total y área total de las ciudades
correlacion=df['population_total'].corr(df['area_total_sq_mi'])
print('--- Correlación entre la población total y el área total de las ciudades ---')
print(correlacion)
# Representamos gráficamente la relación
fig,ax=plt.subplots()
scatter=ax.scatter(df['area_total_sq_mi'], df['population_total'])
ax.set_xlabel('Área total (mi²)')
ax.set_ylabel('Población total')
# Añadimos etiquetas a los puntos
mplcursors.cursor(scatter).connect(
    'add',lambda sel:sel.annotation.set_text(
    f"Ciudad: {df['city'].iloc[sel.target.index]}\nÁrea: {df['area_total_sq_mi'].iloc[sel.target.index]} mi²\nPoblación: {df['population_total'].iloc[sel.target.index]}"
    )
)
plt.show()

# Correlación entre la elevación y la población total
corr=df['elevation_m'].corr(df['population_total'])
print('--- Correlación entre la elevación y la población total de las ciudades ---')
print(corr)
# Representamos gráficamente la relación
fig,ax=plt.subplots()
scatter=ax.scatter(df['population_total'], df['elevation_m'])
ax.set_xlabel('Población total')
ax.set_ylabel('Elevación (metros)')
# Añadimos etiquetas a los puntos
mplcursors.cursor(scatter).connect(
    'add',lambda sel:sel.annotation.set_text(
    f"Ciudad: {df['city'].iloc[sel.target.index]}\nElevación: {df['elevation_m'].iloc[sel.target.index]} mi²\nPoblación: {df['population_total'].iloc[sel.target.index]}"
    )
)
plt.show()