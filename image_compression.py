# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:00:43 2026

@author: antwi
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from Svd_algorithms import svd_compressor_main

image = io.imread('original_images\Dr.Zhao.jpg')
gray = color.rgb2gray(image)

A = gray

# Original SVD Code: U, S, VT = np.linalg.svd(A, full_matrices=False)
S_matrix, U, V = svd_compressor_main(A)

# 1. Extract the 1D array for S (most compression logic expects this)
S = np.diag(S_matrix) 

# 2. Transpose V to get VT
VT = V.T

k = 35 #reduce k makes blurry

A_k = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

# Plot 
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(A, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title(f"Compressed (k={k})")
plt.imshow(A_k, cmap='gray')
plt.axis('off')

plt.show()