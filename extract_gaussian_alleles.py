import pandas as pd
import numpy as np
from sklearn import mixture
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def extract_gaussian_alleles(distances, n_components=None):
    data = np.array(distances).reshape(-1, 1)
    
    # remove outliers
    nearest, indices = NearestNeighbors(n_neighbors=1).fit(data).kneighbors()
    nearest = np.sort(nearest, axis=0)[:, 0]
    knee = KneeLocator(range(len(nearest)), nearest, curve='convex').knee_y
    eps = max(250, knee)
    min_samples = max(2, int(len(distances) * 0.1))
    
    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)
    keep = clusters >= 0

    data = np.array(distances[keep]).reshape(-1, 1)

    if n_components:
        n_min = n_components
        n_max = n_components
    else:
        n_min = max(2, clusters.max() + 1)
        n_max = min(4, len(data))
        n_max = max(n_max, n_min)

    lowest_bic = np.infty
    model = None
    for n in range(n_min, n_max + 1):
        m = mixture.GaussianMixture(n).fit(data)
        bic = m.bic(data)
        if bic < lowest_bic:
            lowest_bic = bic
            lowest_bic_model = m
            if all(m.weights_ > 0.15):
                model = m
    if not model:
        model = lowest_bic_model

    means = model.means_[:, 0]
    stds = model.covariances_[:, 0, 0] ** 0.5
    weights = model.weights_

    df = pd.DataFrame(zip(means, stds, weights), columns=['mean', 'std', 'weight']).sort_values('mean').T
    df.columns = list(range(len(df.columns)))
    df = df.stack()

    return df, model, keep

