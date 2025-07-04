import numpy as np
import json
import os
import time
from typing import List, Tuple, Dict, Any

from geopy.distance import geodesic
import requests
from requests.exceptions import RequestException
from pymcdm.methods import SPOTIS
from pymcdm.weights.subjective.rancom import RANCOM

# Ustawienia numpy
np.set_printoptions(precision=4, suppress=True, linewidth=1000)


def count(lat: float, lon: float, tag_key: str, tag_value: str, 
                    radius: int = 500, max_retries: int = 5, delay: float = 1.0) -> int:
    """
    Zlicza punkty zainteresowania (POI) w okolicy zadanego punktu.
    
    Używa Overpass API do zapytań do bazy danych OpenStreetMap. Funkcja implementuje
    mechanizm retry w przypadku błędów sieciowych lub rate limitingu.
    
    Args:
        lat (float): Szerokość geograficzna punktu
        lon (float): Długość geograficzna punktu
        tag_key (str): Klucz tagu OSM (np. "amenity", "shop")
        tag_value (str): Wartość tagu OSM (np. "restaurant", "supermarket")
        radius (int, optional): Promień wyszukiwania w metrach. Defaults to 500.
        max_retries (int, optional): Maksymalna liczba prób. Defaults to 5.
        delay (float, optional): Opóźnienie między zapytaniami w sekundach. Defaults to 1.0.
    
    Returns:
        int: Liczba znalezionych POI w promieniu lub 0 w przypadku błędu
        
    Example:
        >>> count(53.4329, 14.5479, "amenity", "restaurant", radius=1000)
        15
    """
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            
            overpass_url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json][timeout:25];
            (
              node["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
              way["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
              relation["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
            );
            out count;
            """

            response = requests.get(overpass_url, params={'data': query}, timeout=30)
            
            if response.status_code == 429:
                wait_time = 2 ** attempt * 5
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()

            if (data.get('elements') and 
                data['elements'][0].get('tags', {}).get('total')):
                return int(data['elements'][0]['tags']['total'])
            else:
                return 0
                
        except RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return 0
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 0

    return 0


def find_nearest_poi(lat: float, lon: float, tag_key: str, tag_value: str,
                               radius: int = 500, max_retries: int = 5, delay: float = 1.0) -> float:
    """
    Znajduje odległość do najbliższego punktu zainteresowania (POI) określonego typu.
    
    Używa Overpass API do pobrania wszystkich POI w określonym promieniu, następnie
    oblicza odległość geodezyjną do każdego i zwraca najmniejszą odległość.
    Implementuje mechanizm retry w przypadku błędów sieciowych.
    
    Args:
        lat (float): Szerokość geograficzna punktu startowego
        lon (float): Długość geograficzna punktu startowego  
        tag_key (str): Klucz tagu OSM (np. "highway", "amenity")
        tag_value (str): Wartość tagu OSM (np. "cycleway", "hospital")
        radius (int, optional): Maksymalny promień wyszukiwania w metrach. Defaults to 500.
        max_retries (int, optional): Maksymalna liczba prób w przypadku błędu. Defaults to 5.
        delay (float, optional): Opóźnienie między zapytaniami w sekundach. Defaults to 1.0.
    
    Returns:
        float: Odległość w metrach do najbliższego POI lub wartość radius jeśli 
               nic nie znaleziono lub wystąpił błąd
               
    Example:
        >>> find_nearest_poi(53.4329, 14.5479, "highway", "cycleway", radius=1000)
        245.7
    """
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            
            url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json][timeout:25];
            (
              node["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
              way["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
              relation["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
            );
            out center;
            """
            
            response = requests.get(url, params={'data': query}, timeout=30)
            
            if response.status_code == 429:
                wait_time = 2 ** attempt * 5
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            min_dist = float('inf')
            for el in data.get('elements', []):
                poi_coord = None
                if el['type'] == 'node':
                    poi_coord = (el['lat'], el['lon'])
                elif 'center' in el:
                    poi_coord = (el['center']['lat'], el['center']['lon'])
                else:
                    continue
                    
                dist = geodesic((lat, lon), poi_coord).meters
                if dist < min_dist:
                    min_dist = dist
            
            return min_dist if min_dist < float('inf') else radius
            
        except RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return radius
        except Exception as e:
            print(f"Unexpected error: {e}")
            return radius
    
    return radius


def save_cache(data: Dict, filename: str = 'cache.json') -> None:
    """
    Zapisuje dane do pliku cache w formacie JSON.
    
    Args:
        data (Dict): Słownik z danymi do zapisania
        filename (str, optional): Nazwa pliku cache. Defaults to 'cache.json'.
    
    Returns:
        None
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_cache(filename: str = 'cache.json') -> Dict:
    """
    Wczytuje dane z pliku cache w formacie JSON.
    
    Args:
        filename (str, optional): Nazwa pliku cache do wczytania. Defaults to 'cache.json'.
    
    Returns:
        Dict: Słownik z wczytanymi danymi lub pusty słownik jeśli plik nie istnieje
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}



def get_criteria_vector(point: Tuple[float, float], criteria_list: List[Dict],
                       radius: int = 500, delay: float = 1.0) -> List[float]:
    """
    Oblicza wektor kryteriów dla pojedynczego punktu geograficznego.
    
    Dla każdego kryterium z listy wywołuje odpowiednią funkcję (count lub find_nearest_poi)
    i zwraca wektor wartości numerycznych reprezentujących wszystkie kryteria.
    
    Args:
        point (Tuple[float, float]): Współrzędne punktu (szerokość, długość geograficzna)
        criteria_list (List[Dict]): Lista słowników definiujących kryteria z kluczami:
                                   - "method": "count" lub "distance"
                                   - "api_params": tuple (tag_key, tag_value)
                                   - "name": nazwa kryterium
        radius (int, optional): Promień wyszukiwania w metrach. Defaults to 500.
        delay (float, optional): Opóźnienie między zapytaniami API. Defaults to 1.0.
    
    Returns:
        List[float]: Wektor wartości kryteriów dla danego punktu
    
    Example:
        >>> criteria = [{"method": "count", "api_params": ("amenity", "restaurant"), "name": "Restauracje"}]
        >>> get_criteria_vector((53.4329, 14.5479), criteria)
        [12.0]
    """
    lat, lon = point
    values = []
    
    for crit_def in criteria_list:
        tag_key, tag_val = crit_def["api_params"]
        method = crit_def["method"]
        
        try:
            if method == "distance":
                value = find_nearest_poi(lat, lon, tag_key, tag_val, radius, delay=delay)
            elif method == "count":
                value = float(count(lat, lon, tag_key, tag_val, radius, delay=delay))
            else:
                value = 0.0
            values.append(value)
        except Exception as e:
            print(f"Error processing {crit_def['name']} for point ({lat}, {lon}): {e}")
            values.append(0.0)
    
    return values


def point_to_key(point: Tuple[float, float]) -> str:
    """
    Konwertuje punkt geograficzny na klucz string dla mechanizmu cache.
    
    Formatuje współrzędne z dokładnością do 6 miejsc po przecinku, co zapewnia
    dokładność około 11 cm na poziomie morza.
    
    Args:
        point (Tuple[float, float]): Współrzędne punktu (szerokość, długość geograficzna)
    
    Returns:
        str: Klucz w formacie "lat,lon" z 6 miejscami po przecinku
        
    Example:
        >>> point_to_key((53.432961, 14.547969))
        "53.432961,14.547969"
    """
    return f"{point[0]:.6f},{point[1]:.6f}"


def process_points_in_chunks(points_list: List[Tuple[float, float]], 
                           criteria_list: List[Dict],
                           chunk_size: int = 10,
                           radius: int = 500,
                           delay: float = 1.0,
                           chunk_delay: float = 5.0,
                           cache_file: str = 'cache.json') -> np.ndarray:
    """
    Przetwarza listę punktów geograficznych w małych fragmentach (chunkach) z mechanizmem cache'owania.
    
    Funkcja dzieli dużą listę punktów na mniejsze fragmenty, aby uniknąć przeciążenia API
    i umożliwić regularne zapisywanie postępu. Używa mechanizmu cache aby nie powtarzać
    już obliczonych wartości.
    
    Args:
        points_list (List[Tuple[float, float]]): Lista punktów do przetworzenia
        criteria_list (List[Dict]): Lista definicji kryteriów do obliczenia
        chunk_size (int, optional): Rozmiar fragmentu punktów. Defaults to 10.
        radius (int, optional): Promień wyszukiwania w metrach. Defaults to 500.
        delay (float, optional): Opóźnienie między zapytaniami API. Defaults to 1.0.
        chunk_delay (float, optional): Opóźnienie między fragmentami w sekundach. Defaults to 5.0.
        cache_file (str, optional): Nazwa pliku cache. Defaults to 'cache.json'.
    
    Returns:
        np.ndarray: Macierz 2D gdzie każdy wiersz reprezentuje punkt, a kolumny to wartości kryteriów
        
    Example:
        >>> points = [(53.4329, 14.5479), (53.4350, 14.5500)]
        >>> criteria = [{"method": "count", "api_params": ("amenity", "restaurant"), "name": "Restauracje"}]
        >>> matrix = process_points_in_chunks(points, criteria, chunk_size=2)
        >>> matrix.shape
        (2, 1)
    """
    cache = load_cache(cache_file)
    all_results = []
    
    total_chunks = (len(points_list) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(0, len(points_list), chunk_size):
        chunk = points_list[chunk_idx:chunk_idx + chunk_size]
        chunk_num = chunk_idx // chunk_size + 1
        
        print(f"Processing chunk {chunk_num}/{total_chunks}")
        
        chunk_results = []
        for i, point in enumerate(chunk):
            key = point_to_key(point)
            
            if key in cache:
                chunk_results.append(cache[key])
            else:
                print(f"Processing point {chunk_idx + i + 1}: {point}")
                try:
                    result = get_criteria_vector(point, criteria_list, radius, delay)
                    chunk_results.append(result)
                    cache[key] = result
                    save_cache(cache, cache_file)
                except Exception as e:
                    print(f"Error processing point {point}: {e}")
                    default_result = [0.0] * len(criteria_list)
                    chunk_results.append(default_result)
                    cache[key] = default_result
        
        all_results.extend(chunk_results)
        
        if chunk_num < total_chunks:
            time.sleep(chunk_delay)
    
    return np.array(all_results)


# def generate_grid_points(lat_range: Tuple[float, float], 
#                         lon_range: Tuple[float, float], 
#                         grid_size: int = 10) -> List[Tuple[float, float]]:
#     """
#     Generuje regularną siatkę punktów geograficznych do analizy.
    
#     Tworzy równomiernie rozłożone punkty w prostokątnym obszarze geograficznym.
#     Użyteczne do systematycznej analizy całego regionu zamiast konkretnych lokalizacji.
    
#     Args:
#         lat_range (Tuple[float, float]): Zakres szerokości geograficznej (min, max)
#         lon_range (Tuple[float, float]): Zakres długości geograficznej (min, max)
#         grid_size (int, optional): Liczba punktów na każdej osi. Defaults to 10.
#                                   Całkowita liczba punktów = grid_size²
    
#     Returns:
#         List[Tuple[float, float]]: Lista współrzędnych punktów (lat, lon)
        
#     Example:
#         >>> points = generate_grid_points((53.40, 53.45), (14.50, 14.55), 3)
#         >>> len(points)
#         9
#         >>> points[0]
#         (53.40, 14.50)
#     """
#     lat_grid = np.linspace(lat_range[0], lat_range[1], grid_size)
#     lon_grid = np.linspace(lon_range[0], lon_range[1], grid_size)
    
#     lon0, lat0 = np.meshgrid(lon_grid, lat_grid)
#     pts = np.column_stack([lat0.ravel(), lon0.ravel()])
    
#     return [(float(point[0]), float(point[1])) for point in pts]


def merge_poi_criteria(alts: np.ndarray, poi_indices: List[int]) -> np.ndarray:
    """
    Merguje wybrane kryteria POI w jedno zagregowane kryterium.
    
    Sumuje wartości wybranych kolumn (kryteriów) i tworzy nową macierz z połączonymi
    kryteriami. Używane do redukcji liczby kryteriów przez grupowanie podobnych konceptów.
    
    Args:
        alts (np.ndarray): Macierz alternatyw z wszystkimi kryteriami
        poi_indices (List[int]): Indeksy kolumn do zsumowania
    
    Returns:
        np.ndarray: Nowa macierz z pierwszymi 3 kryteriami, zsumowanymi POI i ostatnim kryterium
        
    Example:
        >>> alts = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        >>> merge_poi_criteria(alts, [3, 4])
        array([[ 1,  2,  3,  9,  6],
               [ 7,  8,  9, 21, 12]])
    """
    merged_poi = alts[:, poi_indices].sum(axis=1)
    
    return np.column_stack((alts[:, :3], merged_poi, alts[:, -1]))


def calculate_bounds(data: np.ndarray, tolerance: float = 0.1) -> np.ndarray:
    """
    Oblicza bounds (granice) dla metody SPOTIS z dodatkową tolerancją.
    
    Metoda SPOTIS wymaga zdefiniowania idealnego i najgorszego punktu dla każdego kryterium.
    Funkcja oblicza minimalne i maksymalne wartości w danych i dodaje margines tolerancji.
    
    Args:
        data (np.ndarray): Macierz danych z wartościami kryteriów
        tolerance (float, optional): Wartość tolerancji dodawana do granic. Defaults to 0.1.
    
    Returns:
        np.ndarray: Macierz 2D z kolumnami [min_bound, max_bound] dla każdego kryterium
        
    Example:
        >>> data = np.array([[1, 2], [3, 4], [5, 6]])
        >>> calculate_bounds(data, tolerance=0.5)
        array([[0.5, 5.5],
               [1.5, 6.5]])
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    return np.column_stack([
        min_vals - tolerance,
        max_vals + tolerance
    ])


def export_results(preferences: np.ndarray, ranking: np.ndarray, 
                  points_names: List[str],
                  output_prefix: str = 'results') -> None:
    """
    Eksportuje wyniki analizy do plików w formatach CSV i JSON.
    
    Tworzy pliki z preferencjami (wartości numeryczne) oraz ranking z nazwami punktów.
    Umożliwia dalszą analizę i wizualizację wyników.
    
    Args:
        preferences (np.ndarray): Tablica wartości preferencji dla każdego punktu
        ranking (np.ndarray): Tablica z rankingiem punktów (1-indexed)
        points_names (List[str]): Lista nazw punktów odpowiadająca rankingowi
        output_prefix (str, optional): Prefix nazw plików wyjściowych. Defaults to 'results'.
    
    Returns:
        None
        
    Creates files:
        - {output_prefix}_preferences.csv: Wartości preferencji w formacie CSV
        - {output_prefix}_preferences.json: Wartości preferencji w formacie JSON
        - {output_prefix}_ranking.json: Kompletny ranking z nazwami i preferencjami
    """
    np.savetxt(f'{output_prefix}_preferences.csv', preferences, delimiter=',', 
               header='preference', comments='')
    
    with open(f'{output_prefix}_preferences.json', 'w') as f:
        json.dump(preferences.tolist(), f, indent=2)
    
    ranking_data = {
        'ranking': ranking.tolist(),
        'points': points_names,
        'preferences': preferences.tolist()
    }
    
    with open(f'{output_prefix}_ranking.json', 'w') as f:
        json.dump(ranking_data, f, indent=2)


def analyze_locations(points: List[Tuple[float, float]], 
                     points_names: List[str],
                     criteria: List[Dict], 
                     criteria_types: np.ndarray,
                     weights_file: str,
                     poi_indices: List[int] = None,
                     output_prefix: str = 'results',
                     export_results: bool = True,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Główna funkcja analizy lokalizacji
    
    Args:
        points: Lista punktów (lat, lon) do analizy
        points_names: Nazwy punktów
        criteria: Lista definicji kryteriów
        criteria_types: Tablica typów kryteriów (1 = więcej lepiej, -1 = mniej lepiej)
        weights_file: Ścieżka do pliku z wagami
        poi_indices: Indeksy kryteriów do mergowania (opcjonalne)
        output_prefix: Prefix dla plików wyjściowych
        **kwargs: Dodatkowe parametry (chunk_size, radius, delay, etc.)
    
    Returns:
        (preferences, ranking)
    """
    
    # Parametry
    chunk_size = kwargs.get('chunk_size', 10)
    radius = kwargs.get('radius', 500)
    delay = kwargs.get('delay', 1.0)
    chunk_delay = kwargs.get('chunk_delay', 5.0)
    cache_file = kwargs.get('cache_file', f'{output_prefix}_cache.json')


    # 1. Przetwarzanie danych
    alts = process_points_in_chunks(points, criteria, chunk_size, radius, delay, chunk_delay, cache_file)
    
    # 2. Mergowanie kryteriów POI (jeśli podano)
    if poi_indices:
        alts_final = merge_poi_criteria(alts, poi_indices)
    else:
        alts_final = alts
    
    # 3. Wczytanie wag
    rancom = RANCOM(filename=weights_file)
    weights = rancom()
    
    # 4. Analiza SPOTIS
    bounds = calculate_bounds(alts_final)
    spotis = SPOTIS(bounds)
    preferences = spotis(alts_final, weights, criteria_types)
    ranking = spotis.rank(preferences)
    
    # 5. Wyświetlenie wyników do naprawy?
    # for i in range(min(3, len(ranking))):
    #     idx = int(ranking[i] - 1)  # ranking is 1-indexed, convert to int
    #     print(f"  {i+1}. {points_names[idx]} (preference: {preferences[idx]:.4f})")
    
    # 6. Eksport wyników
    if export_results:
        export_results(preferences, ranking, points_names, output_prefix)
    
    return preferences, ranking



# if __name__ == "__main__":
#     """
#     Główny skrypt do analizy najlepszych lokalizacji dla infrastruktury rowerowej w Szczecinie.
    
#     Skrypt definiuje kryteria oceny, punkty do analizy i przeprowadza wielokryterialną analizę
#     decyzyjną używając metody SPOTIS do znalezienia najlepszych lokalizacji dla infrastruktury
#     rowerowej. Wyniki są eksportowane do plików JSON i CSV.
    
#     Kryteria analizy obejmują:
#     - Dostępność komunikacji publicznej
#     - Bliskość uniwersytetów
#     - Istniejące ścieżki rowerowe
#     - Dostępność usług (supermarkety, restauracje, biura)
#     - Atrakcje turystyczne i rekreacyjne
#     - Gęstość zabudowy mieszkaniowej
    
#     Można analizować konkretne punkty (use_grid=False) lub regularną siatkę punktów (use_grid=True).
#     """
    
#     BIKE_CRITERIA = [
#         {
#             "name": "Ilość komunikacji miejskiej",
#             "id": "count_pub_trans",
#             "method": "count",
#             "api_params": ("public_transport", "platform"),
#             "type": 1,
#         },
#         {
#             "name": "Ilość obszarów uniwersyteckich",
#             "id": "count_uni_area",
#             "method": "count",
#             "api_params": ("amenity", "university"),
#             "type": 1,
#         },
#         {
#             "name": "Ścieżki rowerowe",
#             "id": "bi_path",
#             "method": "distance",
#             "api_params": ("highway", "cycleway"),
#             "type": -1,
#         },
#         {
#             "name": "Ilość supermarketów",
#             "id": "supermarket",
#             "method": "count",
#             "api_params": ("shop", "supermarket"),
#             "type": 1,
#         },
#         {
#             "name": "Ilość parków w okolicy",
#             "id": "recreation_park_count",
#             "method": "count",
#             "api_params": ("leisure", "park"),
#             "type": 1,
#         },
#         {
#             "name": "Ilość biur/miejsc pracy",
#             "id": "office_count",
#             "method": "count",
#             "api_params": ("office", "yes"),
#             "type": 1,
#         },
#         {
#             "name": "Ilość restauracji",
#             "id": "restaurant_count",
#             "method": "count",
#             "api_params": ("amenity", "restaurant"),
#             "type": 1,
#         },
#         {
#             "name": "Ilość fast foodów",
#             "id": "fast_food_count",
#             "method": "count",
#             "api_params": ("amenity", "fast_food"),
#             "type": 1,
#         },
#         {
#             "name": "Ilość atrakcji turystycznych",
#             "id": "tourism_attraction_count",
#             "method": "count",
#             "api_params": ("tourism", "attraction"),
#             "type": 1,
#         },
#         {
#             "name": "Gęstość zabudowy mieszkaniowej",
#             "id": "residential_building_count",
#             "method": "count",
#             "api_params": ("building", "residential"),
#             "type": 1,
#         },
#     ]

#     # Punkty w Szczecinie
#     BIKE_POINTS = [
#         (53.43296129639522, 14.547968868949667),  # Plac Grunwaldzki
#         (53.43209966711944, 14.555278766434393),  # Pazim/Galaxy
#         (53.44771696811074, 14.49176316404754),   # WI
#         (53.42733614288465, 14.485328899575466),  # Ster
#         (53.42777787163157, 14.53133731478859),   # Turzyn
#         (53.40365605376617, 14.499642240087212),  # Cukrowa Uniwerek
#         (53.44247491754028, 14.567364906386535),  # Fabryka Wody + Mieszkania
#         (53.427400712136276, 14.537395453240668), # Kościuszki
#         (53.422574981749726, 14.559356706858086), # Wyszyńskiego
#         (53.42451975173007, 14.550517853490351),  # Brama Portowa
#     ]

#     BIKE_POINTS_NAMES = [
#         "Plac Grunwaldzki", "Pazim/Galaxy", "WI", "Ster", "Turzyn",
#         "Cukrowa Uniwerek", "Fabryka Wody + Mieszkania", "Kościuszki",
#         "Wyszyńskiego", "Brama Portowa"
#     ]

#     # Konfiguracja dla rowerów
#     POI_INDICES = [3, 5, 6, 7, 8]  # supermarkety, biura, restauracje, fast food, atrakcje
#     CRITERIA_TYPES = np.array([1, 1, -1, 1, 1])  # po mergowaniu POI
    
#     # Można też analizować siatkę punktów zamiast konkretnych lokalizacji
#     use_grid = True
    
#     if use_grid:
#         analysis_points = generate_grid_points((53.40, 53.45), (14.50, 14.55), 2)
#         analysis_names = [f"Point_{i}" for i in range(len(analysis_points))]
#     else:
#         analysis_points = BIKE_POINTS
#         analysis_names = BIKE_POINTS_NAMES
    
#     # Uruchomienie analizy
#     preferences, ranking = analyze_locations(
#         points=analysis_points,
#         points_names=analysis_names,
#         criteria=BIKE_CRITERIA,
#         criteria_types=CRITERIA_TYPES,
#         weights_file="as_rancom.csv",
#         poi_indices=POI_INDICES,
#         output_prefix="bike_analysis",
#         chunk_size=10,
#         radius=500,
#         delay=1.0,
#         chunk_delay=5.0
#     )
    
