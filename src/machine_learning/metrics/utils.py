import numpy as np

def R_squared_models(models):
    mean_model = np.mean(models, axis=0)
    SSR, SST = 0, 0
    for i in range(len(models)):
        norm = np.linalg.norm(models[i])
        SST += norm**2

        delta = models[i] - mean_model
        norm = np.linalg.norm(delta)
        SSR += norm**2

    R_squared = 1 - SSR/SST
    return R_squared