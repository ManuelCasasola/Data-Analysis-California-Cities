import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn

# Accedemos al archivo y comprobamos la información
ciudades=pd.read_csv('california_cities.csv')
print(ciudades.head())
print(ciudades.describe())

# Extraemos la información en la que estamos interesados
latitud,longitud=ciudades['latd'],ciudades['longd']
poblacion,area=ciudades['population_total'],ciudades['area_total_km2']

# Ahora realizaremos el gráfico
seaborn.set()
plt.scatter(longitud,latitud,label=None,c=np.log10(poblacion),
            cmap='viridis',s=area,linewidths=0,alpha=0.5)
plt.gca().set_aspect('equal')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.colorbar(label='log$_{10}$(poblacion)')
plt.clim(3,7)

for area in [100,300,500]:
    plt.scatter([],[],c='k',alpha=0.3,s=area,label=str(area)+'km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Areas Ciudades')
plt.title("Área y Población de las ciudades de California")
plt.show()