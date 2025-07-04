import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import griddata


def interpolate(pts, vals, latitudes, longitudes, interpolation_factor=10):
    """
    Interpoluje wartości na siatce geograficznej.
    pts - punkty wejściowe (lon, lat),
    vals - wartości w punktach,
    latitudes, longitudes - siatki szerokości i długości,
    interpolation_factor - zagęszczenie siatki.
    Zwraca: siatkę długości, szerokości oraz interpolowaną macierz wartości.
    """
    n_lat = int(interpolation_factor * len(latitudes))
    n_lon = int(interpolation_factor * len(longitudes))
    
    lat_i = np.linspace(latitudes.min(), latitudes.max(), n_lat)
    lon_i = np.linspace(longitudes.min(), longitudes.max(), n_lon)
    lon_i_m, lat_i_m = np.meshgrid(lon_i, lat_i)

    vals = np.array(vals).ravel()

    # RBF interpolacja - do nieregularnych punktów (np. autobusy)
    # query = np.column_stack([lon_i_m.ravel(), lat_i_m.ravel()])
    # rbf = RBFInterpolator(pts, vals, kernel='thin_plate_spline')
    # grid = rbf(query).reshape(lat_i_m.shape)

    # Przy regularnej siatce używamy griddata (wydajniej)
    grid = griddata(
        pts, vals,
        (lon_i_m, lat_i_m),
        method='cubic'
    )

    return lon_i_m, lat_i_m, grid

