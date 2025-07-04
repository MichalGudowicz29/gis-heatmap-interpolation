import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

def plot_heatmap(grid, lon_i_m, lat_i_m, title, alpha=0.6, osm_zoom=14, equal_degree_aspect=True):
    """
    Rysuje mapę cieplną (heatmapę) na podkładzie OSM.
    grid - wartości do wyświetlenia,
    lon_i_m, lat_i_m - siatka długości i szerokości,
    title - tytuł wykresu.
    """
    # Dodaj podkład OSM
    osm = cimgt.OSM()
    extent = [lon_i_m.min(), lon_i_m.max(), lat_i_m.min(), lat_i_m.max()]
    
    # Pierwszy wykres: sama mapa OSM
    fig1, ax1 = plt.subplots(
        figsize=(10, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    ax1.add_image(osm, osm_zoom)
    ax1.set_extent(extent, crs=ccrs.PlateCarree())
    ax1.set_title(f'{title} - OSM Basemap Only')
    
    if equal_degree_aspect:
        ax1.set_aspect('equal', adjustable='box')

    # Drugi wykres: OSM + heatmapa
    fig2, ax2 = plt.subplots(
        figsize=(10, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    ax2.add_image(osm, osm_zoom)
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    ax2.set_title(f'{title} - Heatmap Overlay')

    pcm = ax2.pcolormesh(
        lon_i_m, lat_i_m, grid,
        cmap='coolwarm_r',
        shading='auto',
        transform=ccrs.PlateCarree(),
        alpha=alpha
    )

    # Pasek kolorów
    cbar = fig2.colorbar(pcm, ax=ax2, orientation='vertical', pad=0.05, aspect=20)
    cbar.set_label('lower = better')
    
    if equal_degree_aspect:
        ax2.set_aspect('equal', adjustable='box')

    plt.show()

def plot_points(points, lat_bounds, lon_bounds, title="Points on OSM", osm_zoom=14):
    """
    Rysuje punkty na mapie OSM.
    points - tablica punktów (lon, lat),
    lat_bounds, lon_bounds - zakresy szerokości i długości,
    title - tytuł wykresu.
    """
    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # Dodaj tło OSM
    osm = cimgt.OSM()
    ax.add_image(osm, osm_zoom)
    
    # Ustaw zakres mapy
    ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], 
                  crs=ccrs.PlateCarree())
    
    # Rysuj punkty
    lons = points[:, 0]
    lats = points[:, 1]
    
    ax.scatter(lons, lats, 
               c='red', s=50, alpha=0.7, 
               transform=ccrs.PlateCarree(),
               edgecolors='black', linewidths=0.5,
               label=f'{len(points)} points')
    
    # Dodaj siatkę i opisy
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    

def generate_points(lat0, lat1, lon0, lon1, distance_per_point):
    """
    Generuje siatkę punktów w zadanym prostokącie geograficznym.
    Zwraca: siatkę szerokości, długości oraz tablicę punktów (lon, lat).
    """
    # Przelicz stopnie na metry (przybliżenie)
    # 1 stopień szerokości ≈ 111 km
    # 1 stopień długości ≈ 111 km * cos(szerokość)
    avg_lat = (lat0 + lat1) / 2
    
    width_in_meters = np.abs(lat1 - lat0) * 111000  # metry na stopień szerokości
    height_in_meters = np.abs(lon1 - lon0) * 111000 * np.cos(np.radians(avg_lat))  # metry na stopień długości

    w_points = int(width_in_meters / distance_per_point)
    h_points = int(height_in_meters / distance_per_point)

    # Minimum 2 punkty w każdym kierunku
    w_points = max(2, w_points)
    h_points = max(2, h_points)

    lat_grid = np.linspace(lat0, lat1, w_points)
    lon_grid = np.linspace(lon0, lon1, h_points)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    pts = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
    
    print(f"Generated grid: {w_points} x {h_points} = {w_points * h_points} points")
    
    return lat_grid, lon_grid, pts

def summary(latitudes,longitudes, lat0, lat1, lon0, lon1, points, score):
    """
    Wypisuje podsumowanie wygenerowanych punktów i wyników.
    """
    print(20*"=")
    print(f"SW corner (SW): {lat0}, {lon0}")
    print(f"NE corner (NE): {lat1}, {lon1}")
    print(f"Area: {np.abs(lat0-lat1):.3f} lat x {np.abs(lon0-lon1):.3f} lon")
    print(20*"=")
    print(f"Number of points generated: {len(points)}")
    print(f"Number of score values: {len(score)}")
    print(f"Grid shape: {len(latitudes)} x {len(longitudes)} = {len(latitudes) * len(longitudes)}")
    print(20*"=")
    print(f"Average score in this area: {sum(score)/len(score)}")
