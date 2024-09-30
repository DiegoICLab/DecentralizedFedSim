import numpy as np
import pandas as pd

from typing import List
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# #############################################################################
# Mean
# #############################################################################

def Mean_aggregation(models):
    """Compute average mean"""
    return np.mean(models, axis = 0)

# #############################################################################
# FedAvg
# #############################################################################

def FedAvg_aggregation(models, weights):
    """
    Compute the weighted mean of models
    """
    # Calculate the total number of examples used during training
    # num_examples_total = sum([num_examples for _, num_examples in models])
    weights = np.array(weights)

    # Step 1: Multiply each model by its corresponding weight.
    # This uses broadcasting: weights[:, np.newaxis] changes weights to a column vector.
    weighted_models = models * weights[:, np.newaxis]

    # Step 2: Sum the weighted models.
    sum_weighted_models = np.sum(weighted_models, axis=0)

    # Step 3: Divide by the sum of weights to get the weighted mean.
    total_weight = np.sum(weights)
    weighted_mean = sum_weighted_models / total_weight
    return weighted_mean

# #############################################################################
# Median
# #############################################################################
def Median_aggregation(models):
    """Compute median."""
    return np.median(models, axis = 0)

# #############################################################################
# Trimmed-Mean
# #############################################################################
# Code modified from the Flower project on GitHub
# Repository link: https://github.com/adap/flower
def TrimmedMean_aggregation(models, proportiontocut):
    """
    """
    return [
        _trim_mean(np.asarray(layer), proportiontocut=proportiontocut) for layer in zip(*models)
    ]

# Code modified from the Flower project on GitHub
# Repository link: https://github.com/adap/flower
def _trim_mean(array, proportiontocut: float):
    """
    """

    axis = 0
    nobs = array.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut > uppercut:
        raise ValueError("Proportion too big.")

    atmp = np.partition(array, (lowercut, uppercut - 1), axis)

    slice_list = [slice(None)] * atmp.ndim
    slice_list[axis] = slice(lowercut, uppercut)
    result = np.mean(atmp[tuple(slice_list)], axis=axis)
    return result

# #############################################################################
# Krum and Multi-Krum
# #############################################################################

def MultiKrum_aggregation(local_model, neighbor_models, num_malicious, to_keep):

    selected_index = MultiKrum_filtering(neighbor_models, num_malicious, to_keep)
    selected_models = [neighbor_models[i] for i in selected_index]
    if local_model is not None:
        selected_models.append(local_model)
    return Mean_aggregation(selected_models), selected_index

# Code modified from the Flower project on GitHub
# Repository link: https://github.com/adap/flower
def MultiKrum_filtering(models, num_malicious, to_keep):
    """
    Select the relevant models for the MultiKrum algorithm.
    Output: best_results - An array of models (MultiKrum) or a single model (Krum)
    """
    # Compute distances between vectors
    distance_matrix = _compute_distances_matrix(models)

    # For each client, take the n-f-2 closest parameters vectors
    num_closest = max(1, len(models) - num_malicious - 2)
    closest_indices = []
    for i, _ in enumerate(distance_matrix):
        closest_indices.append(
            np.argsort(distance_matrix[i])[1 : num_closest + 1].tolist()  # noqa: E203
        )

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]

    if to_keep > 0:
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = np.argsort(scores)[::-1][len(scores) - to_keep :]  # noqa: E203
        # best_results = [models[i] for i in best_indices]
        return best_indices
    else:
        # Return the model parameters that minimize the score (Krum)
        return [np.argmin(scores)]

# Code modified from the Flower project on GitHub
# Repository link: https://github.com/adap/flower
def _compute_distances_matrix(models):
    """
    Compute matrix distances between vectors.

    Input: 
    - models: list of models vectors
    Output: 
    - distances: matrix distance_matrix of squared distances between the vectors
    """

    distance_matrix = np.zeros((len(models), len(models)))
    for i, _ in enumerate(models):
        for j, _ in enumerate(models):
            distance_matrix[i, j] = compute_euclidean_distance(models[i], models[j])

    return distance_matrix

def compute_euclidean_distance(model_1, model_2):
    """
    Compute Euclidean distance between two vectors.
    """
    delta = model_1 - model_2
    norm = np.linalg.norm(delta)
    euclidean_distance = norm**2
    return euclidean_distance


# #############################################################################
# Clustering and Cosine Similarity
# #############################################################################

def Clustering_aggregation(local_model, neighbor_models):
    """
    Separate the models in two groups by using agglomerative clustering with average link
    based on the cosine distance matrix between each pair of models

    Input:
    - models: list of models vectors

    Output:
    - aggregated_model: resulting model of aggregating the selected models by Clustering algorithm.
    """

    biggest_cluster = Clustering_filtering(neighbor_models)

    if len(biggest_cluster) == 0:
        aggregated_model = local_model
    else:
        selected_models = [neighbor_models[i] for i in biggest_cluster]
        if local_model is not None:
            selected_models.append(local_model)
        aggregated_model = Mean_aggregation(selected_models)
    
    return aggregated_model, biggest_cluster

def Clustering_filtering(models):
    """
    Separate the models in two groups by using agglomerative clustering with average link
    based on the cosine distance matrix between each pair of models

    Input:
    - models: list of models vectors

    Output:
    - aggregated_model: resulting model of aggregating the selected models by Clustering algorithm.
    """
    if len(models) == 1:
        return []
    
    # Compute cosine similarities between vectors
    similarity_matrix = _compute_cosine_similarity_matrix(models)

    # Transform the similarity matrix into a condensed distance matrix necessary in the function linkage
    biggest_cluster = agglomerative_clustering(squareform(similarity_matrix))
    return biggest_cluster

def agglomerative_clustering(agglomerative_distance):
    """
    Agglomerative clustering is a type of hierarchical clustering that treats each data item as an 
    individual cluster initially and then iteratively merges these clusters based on some measure 
    of similarity until a desired number of clusters is reached or some stopping criterion is met. 
    
    The average link is one of the strategies to measure the distance between two clusters. In the 
    average link, the distance between two clusters is calculated as the average of the distances 
    between all pairs of points, where one point belongs to one cluster and the other point belongs 
    to the other cluster. This strategy is useful for finding groups of varying size and shape.
    
    Input:
    - agglomerative_distance: distance matrix between each pair of models

    Output:
    - aggregated_model: resulting model of aggregating the selected models by Clustering algorithm.
    """

    # Apply agglomerative clustering with average link
    Z = linkage(agglomerative_distance, 'average')
    Z = np.abs(Z)

    # Generate two clusters
    # clusters = fcluster(Z, t=max_d, criterion='distance')
    clusters = fcluster(Z, t=2, criterion='maxclust')
    
    # Extracting the indexs of the biggest cluster
    cluster_1 = [i for i, c in enumerate(clusters) if c == 1]
    cluster_2 = [i for i, c in enumerate(clusters) if c == 2]

    return cluster_1 if len(cluster_1) >= len(cluster_2) else cluster_2

def _compute_cosine_similarity_matrix(models):
    """Compute a matrix with the cosine distance/similarity between each pair of models.

    Input: models - list of models vectors
    Output: similarity_matrix - matrix similarity_matrix of cosine distances between the vectors
    """

    similarity_matrix = np.zeros((len(models), len(models)))
    for i, _ in enumerate(models):
        for j, _ in enumerate(models):
            if i==j:
                similarity_matrix[i, j] = 0
            else:
                similarity_matrix[i, j] = compute_cosine_similarity(models[i], models[j])
                test = np.array([similarity_matrix[i, j]])
                nan_indices = np.isnan(test)
                inner_product = np.dot(models[i], models[j])
                similarity_matrix[j, i] = similarity_matrix[i, j]

    return similarity_matrix

def compute_cosine_similarity(model_1, model_2):
    """
    """
    norm_i = np.linalg.norm(model_1)
    norm_j = np.linalg.norm(model_2)
    inner_product = np.dot(model_1, model_2)
    cosine_similarity = 1 - (inner_product / (norm_i * norm_j))
    return cosine_similarity

# #############################################################################
# Proposed algorithms
# #############################################################################

def WFAgg_D_aggregation(local_model, neighbor_models, num_malicious):
    """
    Input:
    - models = neighbors_models + own_model

    Output:
    - aggregated_model: 
    """

    best_indices = WFAgg_D_filtering(neighbor_models, num_malicious)
    best_results = [neighbor_models[i] for i in best_indices]
    if local_model is not None:
        best_results.append(local_model)
    result = Mean_aggregation(best_results)
    return result, best_indices

def WFAgg_D_filtering(models, num_malicious):
    """
    Input:
    Output:
    """
    median = Median_aggregation(models)
    distances_vector = np.zeros((len(models)))

    for i, _ in enumerate(models):
        distances_vector[i] = compute_euclidean_distance(models[i], median)

    num_closest = max(1, len(models) - num_malicious - 1)       # Num.neighbors () - M - 1 = Num.total - M - 2
    best_indices = np.argsort(distances_vector)[0 : num_closest].tolist()
    return best_indices

def WFAgg_C_aggregation(local_model, neighbor_models, num_malicious):
    """
    Input:
    - models = neighbors_models + own_model
    Output:
    - aggregated_model: 
    """

    best_indices = WFAgg_C_filtering(neighbor_models, num_malicious)
    best_results = [neighbor_models[i] for i in best_indices]
    if local_model is not None:
        best_results.append(local_model)
    result = Mean_aggregation(best_results)
    return result, best_indices

def WFAgg_C_filtering(models, num_malicious):
    """
    Input:

    Output:
    - aggregated_model: resulting model of aggregating the selected models by Clustering algorithm.
    """

    median = Median_aggregation(models)
    norm_median = np.linalg.norm(median)
    scaled_models = [ models[i] * np.min([1, norm_median / np.linalg.norm(models[i])], axis = 0) for i in range(len(models))]

    distances_vector = np.zeros((len(models)))

    for i, _ in enumerate(scaled_models):
        distances_vector[i] = compute_cosine_similarity(scaled_models[i], median)

    num_closest = max(1, len(models) - num_malicious - 1)       # Num.neighbors () - M - 1 = Num.total - M - 2
    best_indices = np.argsort(distances_vector)[0 : num_closest].tolist()
    return best_indices

def WFAgg_T_aggregation(local_model, neighbor_models, distance_metrics, cosine_metrics, include_first_models):
    """
    Input:
    - models = neighbors_models + local_model

    Output:
    - aggregated_model: 
    """

    best_indices = WFAgg_T_filtering(distance_metrics, cosine_metrics, include_first_models)
    best_results = [neighbor_models[i] for i in best_indices]
    if local_model is not None:
        best_results.append(local_model)
    result = Mean_aggregation(best_results)
    return result, best_indices

def WFAgg_T_filtering(distance_metrics, cosine_metrics, include_first_models):
    """
    """
    best_indices = []

    for i in range(len(distance_metrics)):
        distances = distance_metrics[i]
        cosines = cosine_metrics[i]

        if (len(distances) <= 2 or len(cosines) <= 2) and include_first_models == True:
            # Not enough information for filtering process
            best_indices.append(i)
            continue
        elif (len(distances) <= 2 or len(cosines) <= 2) and include_first_models == False:
            continue

        current_distance = distances[-1]
        current_cosine = cosines[-1]

        previous_distance = pd.Series(distances[-3:-1][::-1])
        previous_cosine = pd.Series(cosines[-3:-1][::-1])

        # Compute exponential moving average with smoothing factor
        alpha = 0.01

        distance_mean = np.mean(list(previous_distance.ewm(alpha=alpha).mean()), axis = 0)
        cosine_mean = np.mean(list(previous_cosine.ewm(alpha=alpha).mean()), axis = 0)
        distance_std = np.mean(list(previous_distance.ewm(alpha=alpha).std())[1:], axis = 0)
        cosine_std = np.mean(list(previous_cosine.ewm(alpha=alpha).std())[1:], axis = 0)
        
        low_distance = distance_mean - 1.0 * distance_std
        upper_distance = distance_mean + 1.0 * distance_std
        low_cosine = cosine_mean - 1.0 * cosine_std
        upper_cosine = cosine_mean + 1.0 * cosine_std

        if (low_distance <= current_distance and current_distance <= upper_distance
            and low_cosine <= current_cosine and current_cosine <= upper_cosine):

                best_indices.append(i)
    
    return best_indices

def WFAgg_E_aggregation(local_model, neighbor_models, alpha, weights):
    """
    Compute the Exponential Moving Average (EMA) aggregation of local model and neighbors' models.
    This function first calculates a Federated Average (FedAvg) of the neighbors' models using specified weights.
    It then computes the EMA by blending the local model with this weighted average based on the coefficient alpha.

    Inputs:
    - local_model: local node model
    - neighbor_models: A list of model parameters from neighbor nodes
    - alpha: A float between 0 and 1. This is the smoothing factor used in EMA
    - weights: A numpy array representing the weights assigned to each neighbor's model in the averaging process.

    Output:
    - aggregated_model: The aggregated model parameters after applying EMA, in the same structured form as the input models.
    """

    weigthed_neighbors_average = FedAvg_aggregation(neighbor_models, weights)

    if local_model is not None:
        result = (1-alpha) * local_model + alpha * weigthed_neighbors_average
    else:
        result = weigthed_neighbors_average

    return result

def Alt_WFAgg_aggregation(local_model, neighbor_models, num_malicious, smooth_factor, distance_metrics, cosine_metrics):

    best_indices_1 = MultiKrum_filtering(neighbor_models, num_malicious, len(neighbor_models)-3)
    best_indices_2 = Clustering_filtering(neighbor_models)
    best_indices_3 = WFAgg_T_filtering(
        distance_metrics,
        cosine_metrics,
        False
    )

    weights = np.zeros(len(neighbor_models))
    for index in best_indices_1:
        weights[index] += 0.4

    for index in best_indices_2:
        weights[index] += 0.4
            
    for index in best_indices_3:
        weights[index] += 0.2
            
    weights[weights < 0.6] = 0

    if np.all(weights == 0):
        aggregated_model = local_model
    else:
        aggregated_model = WFAgg_E_aggregation(local_model, neighbor_models, smooth_factor, weights)

    return aggregated_model, weights


def WFAgg_aggregation(local_model, neighbor_models, num_malicious, smooth_factor, distance_metrics, cosine_metrics):
    best_indices_1 = WFAgg_D_filtering(neighbor_models, num_malicious)
    best_indices_2 = WFAgg_C_filtering(neighbor_models, num_malicious)
    best_indices_3 = WFAgg_T_filtering(
        distance_metrics,
        cosine_metrics,
        False
    )

    weights = np.zeros(len(neighbor_models))
    for index in best_indices_1:
        weights[index] += 0.4

    for index in best_indices_2:
        weights[index] += 0.4
            
    for index in best_indices_3:
        weights[index] += 0.2
            
    weights[weights < 0.6] = 0

    if np.all(weights == 0):
        aggregated_model = local_model
    else:
        aggregated_model = WFAgg_E_aggregation(local_model, neighbor_models, smooth_factor, weights)

    return aggregated_model, weights