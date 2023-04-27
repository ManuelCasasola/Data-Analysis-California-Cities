import pandas as pd
import folium

# Accedemos al archivo
df=pd.read_csv('california_cities.csv')

# Creamos un mapa centrado en California
map=folium.Map(location=[36.7783, -119.4179], zoom_start=6)

# Agregamos marcadores para cada ciudad
for i, row in df.iterrows():
    popup_text = f"""
    Ciudad: {row['city']}<br>
    Población: {row['population_total']}<br>
    Elevación: {row['elevation_m']} m<br>
    Área: {row['area_total_km2']} km²<br>
    Agua: {row['area_water_percent']}%
    """
    folium.Marker(
        location=[row['latd'],row['longd']],
        popup=popup_text
    ).add_to(map)

# Guardamos el mapa como un archivo HTML
map.save('california_cities_map.html')
