import numpy as np
import torch
import random
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

def comp_rg_eigenvalues(eig_list):
    
    # norm_list = normalize_distributions(eig_list)
    histograms = []    
    for eigenvalues in eig_list:
        eigenvalues = np.histogram(eigenvalues, bins=1, density=True)[0]
        histograms.append(eigenvalues)

    reference_hist = histograms[-1]  # Select the last histogram as the reference

    kl_divergences = []
    for normalized_distribution in histograms[:-1]:
        kl_divergences.append(kl_divergence(reference_hist, normalized_distribution))
        
    kl_divergences_values = []
    for i, kl_divergence_value in enumerate(kl_divergences):
        print('KL Divergence between random distribution and distribution of client', i+1, ':', abs(kl_divergence_value))
        kl_divergences_values.append(abs(kl_divergence_value))
        
    normalized_values = normalize_kl_divergences(kl_divergences_values)
    
    return normalized_values


def comp_avg_eigenvalues(eig_list):
    
    # norm_list = normalize_distributions(eig_list)
    histograms = [] 
    
    for eigenvalues in eig_list:
        eigenvalues = np.histogram(eigenvalues, bins=1, density=True)[0]
        histograms.append(eigenvalues)
    
    avg_distribution = np.mean(histograms[:-1], axis=0)
    
    # Check if avg_distribution has zeros
    if np.any(avg_distribution == 0):
        # Apply Laplace smoothing only to elements where avg_distribution is zero
        epsilon = 1e-10  
        avg_distribution_smoothed = np.where(avg_distribution == 0, epsilon, avg_distribution)
    else:
        avg_distribution_smoothed = avg_distribution
    
    kl_divergences = []
    for normalized_distribution in histograms[:-1]:
        kl_divergences.append(kl_divergence(avg_distribution_smoothed, normalized_distribution))
        
    kl_divergences_values = []
    for i, kl_divergence_value in enumerate(kl_divergences):
        # print('KL Divergence between average distribution and distribution of client', i+1, ':', kl_divergence_value)
        kl_divergences_values.append(kl_divergence_value)
        
    normalized_values = normalize_kl_divergences(kl_divergences_values)        
    return normalized_values

def remove_outliers(dist):
    lower_percentile = int(len(dist) * 0.1)
    upper_percentile = int(len(dist) * 0.9)
    
    trimmed_dist = dist[lower_percentile:upper_percentile]
    return trimmed_dist / np.sum(trimmed_dist)

def normalize_distributions(dist_list):
    normalized_distributions = []
    for dist in dist_list:
        normalized_dist = remove_outliers(dist)
        normalized_distributions.append(normalized_dist)
    return normalized_distributions

def normalize_kl_divergences(kl_divergences_values):
    max_kl = max(kl_divergences_values)
    min_kl = 0

    range_kl = max_kl - min_kl

    normalized_kl_divergences = []

    for kl_divergence_value in kl_divergences_values:
        normalized_value = (kl_divergence_value - min_kl) / range_kl if range_kl != 0 else 0.5
        normalized_kl_divergences.append(normalized_value)

    return normalized_kl_divergences

def kl_divergence(p, q):
    epsilon = 1e-10
    p_safe = p + epsilon
    q_safe = q + epsilon
    return np.sum(p_safe * np.log(p_safe / q_safe))

def client_ratio(kl_divergences_values):
    client_ratios = []
    for i, kl_divergence_value in enumerate(kl_divergences_values):
        client_ratios.append(kl_divergence_value / sum(kl_divergences_values))
        print('Client', i+1, 'ratio:', client_ratios[i])
    return client_ratios
