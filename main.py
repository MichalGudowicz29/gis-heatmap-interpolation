from map_functions import plot_heatmap, plot_points, generate_points, summary
from math_functions import interpolate
from mcda_functions import analyze_locations
import numpy as np

# # Konfiguracja
# Wspolrzedne
lat0 = 53.401 
lat1 = 53.445 
lon0 = 14.508
lon1 = 14.587
# Odstep miedzy punktami
distance_per_point = 500

BIKE_CRITERIA = [
    {
        "name": "Ilość komunikacji miejskiej",
        "id": "count_pub_trans",
        "method": "count",
        "api_params": ("public_transport", "platform"),
        "type": 1,
    },
    {
        "name": "Ilość obszarów uniwersyteckich",
        "id": "count_uni_area",
        "method": "count",
        "api_params": ("amenity", "university"),
        "type": 1,
    },
    {
        "name": "Ścieżki rowerowe",
        "id": "bi_path",
        "method": "distance",
        "api_params": ("highway", "cycleway"),
        "type": -1,
    },
    {
        "name": "Ilość supermarketów",
        "id": "supermarket",
        "method": "count",
        "api_params": ("shop", "supermarket"),
        "type": 1,
    },
    {
        "name": "Ilość parków w okolicy",
        "id": "recreation_park_count",
        "method": "count",
        "api_params": ("leisure", "park"),
        "type": 1,
    },
    {
        "name": "Ilość biur/miejsc pracy",
        "id": "office_count",
        "method": "count",
        "api_params": ("office", "yes"),
        "type": 1,
    },
    {
        "name": "Ilość restauracji",
        "id": "restaurant_count",
        "method": "count",
        "api_params": ("amenity", "restaurant"),
        "type": 1,
    },
    {
        "name": "Ilość fast foodów",
        "id": "fast_food_count",
        "method": "count",
        "api_params": ("amenity", "fast_food"),
        "type": 1,
    },
    {
        "name": "Ilość atrakcji turystycznych",
        "id": "tourism_attraction_count",
        "method": "count",
        "api_params": ("tourism", "attraction"),
        "type": 1,
    },
    {
        "name": "Gęstość zabudowy mieszkaniowej",
        "id": "residential_building_count",
        "method": "count",
        "api_params": ("building", "residential"),
        "type": 1,
    },
]

# Punkty w Szczecinie
BIKE_POINTS = [
    (53.43296129639522, 14.547968868949667),  # Plac Grunwaldzki
    (53.43209966711944, 14.555278766434393),  # Pazim/Galaxy
    (53.44771696811074, 14.49176316404754),   # WI
    (53.42733614288465, 14.485328899575466),  # Ster
    (53.42777787163157, 14.53133731478859),   # Turzyn
    (53.40365605376617, 14.499642240087212),  # Cukrowa Uniwerek
    (53.44247491754028, 14.567364906386535),  # Fabryka Wody + Mieszkania
    (53.427400712136276, 14.537395453240668), # Kościuszki
    (53.422574981749726, 14.559356706858086), # Wyszyńskiego
    (53.42451975173007, 14.550517853490351),  # Brama Portowa
]

BIKE_POINTS_NAMES = [
    "Plac Grunwaldzki", "Pazim/Galaxy", "WI", "Ster", "Turzyn",
    "Cukrowa Uniwerek", "Fabryka Wody + Mieszkania", "Kościuszki",
    "Wyszyńskiego", "Brama Portowa"
]

# Konfiguracja dla rowerów
POI_INDICES = [3, 5, 6, 7, 8]  # supermarkety, biura, restauracje, fast food, atrakcje
CRITERIA_TYPES = np.array([1, 1, -1, 1, 1])  # po mergowaniu POI



def main():
    # use_grid = False

    latitudes, longitudes, points = generate_points(lat0, lat1, lon0, lon1, distance_per_point)
    analysis_points = [(float(point[1]), float(point[0])) for point in points]
    analysis_names = [f"Point_{i}" for i in range(len(analysis_points))]

    # if use_grid:
    #     latitudes, longitudes, points = generate_points(lat0, lat1, lon0, lon1, distance_per_point)
    #     analysis_points = [(float(point[0]), float(point[1])) for point in points]
    #     analysis_names = [f"Point_{i}" for i in range(len(analysis_points))]
    # else:
    #     analysis_points = BIKE_POINTS
    #     analysis_names = BIKE_POINTS_NAMES


    # Generujemy zbiór punktow rozmieszczonych o zadana wartosc w metrach, zbiór szerokosci oraz wysokosci geograficznych
    # latitudes, longitudes, points = generate_points(lat0, lat1, lon0, lon1, distance_per_point)
    # score = spotis_base[:len(points)]

    preferences, ranking = analyze_locations(
        points=analysis_points,
        points_names=analysis_names,
        criteria=BIKE_CRITERIA,
        criteria_types=CRITERIA_TYPES,
        weights_file="as_rancom.csv",
        poi_indices=POI_INDICES,
        output_prefix="bike_analysis",
        export_results=False,
        chunk_size=10,
        radius=500,
        delay=1.0,
        chunk_delay=5.0
    )

    preferences = preferences.tolist()
    
    # generujemy interpolowany grid poszerzony o zadany interpolate_factor (domyslnie 10)
    i_mesh_longitude, i_mesh_latitude, grid = interpolate(points, preferences, latitudes, longitudes)

    # plotowanie heatmapy 
    plot_heatmap(grid, i_mesh_longitude, i_mesh_latitude, "Heatmap")
    
    # plotowanie mapy punktow dla upewnienia sie co do jakosci funkcji ktora je rozmieszcza
    plot_points(points, (lat0, lat1), (lon0, lon1), "Grid Points")

    # podsumowanie wynikow i analizy  
    summary(latitudes,longitudes, lat0,lat1,lon0,lon1,points,preferences)

if __name__ == "__main__":
    main()
