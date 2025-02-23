import numpy as np
import random

def compute_homography(points1, points2):
    # Calcola la matrice di omografia H che trasforma points1 (ista di punti x-y nella prima immagine) in points2 (lista di punti x-y corrispondenti nella seconda immagine)

    A = [] # Matrice dei coefficienti per il calcolo dell'omografia
    for (x1, y1), (x2, y2) in zip(points1, points2):  # Equazioni lineari per ogni coppia di punti
        A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
    
    A = np.array(A) # Decomposizione ai valori singolari SVD

    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)  # L'ultima riga di Vt fornisce la soluzione
    return H / H[2, 2]  # Normalizzazione per rendere H[2,2] = 1


def transform_point(H, point): # Applica la trasformazione omografica H a un punto
    x, y = point
    vec = np.array([x, y, 1])
    transformed_vec = H @ vec
    return transformed_vec[:2] / transformed_vec[2] # Normalizzazione

def compute_inliers(H, points1, points2, threshold): # Conta gli inlier tra points1 e points2 in base alla trasformazione H
    inliers = []
    for i, (p1, p2) in enumerate(zip(points1, points2)):
        projected_p1 = transform_point(H, p1)
        error = np.linalg.norm(projected_p1 - p2)
        if error < threshold: # Soglia di errore per considerare un punto un inlier
            inliers.append(i)
    return inliers

def ransac_homography(points1, points2, num_iterations=500, threshold=5.0): # Implementazione RANSAC per trovare la migliore omografia
    best_H = None
    best_inliers = []
    
    for _ in range(num_iterations):   # Seleziona casualmente 4 punti per calcolare H
        sample_indices = random.sample(range(len(points1)), 4)
        sampled_points1 = [points1[i] for i in sample_indices]
        sampled_points2 = [points2[i] for i in sample_indices]
        
        H = compute_homography(sampled_points1, sampled_points2)
        inliers = compute_inliers(H, points1, points2, threshold)
        
        if len(inliers) > len(best_inliers): # Se ho più inlier aggiorno il valore, altrimenti no
            best_H = H
            best_inliers = inliers
    
    return best_H
