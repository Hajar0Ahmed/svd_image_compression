# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:00:43 2026

@author: antwi
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

image = io.imread('D:\MSU\computalgebra\project-svd\svd_image_compression\original_images')
gray = color.rgb2gray(image)

A = gray

U, S, VT = np.linalg.svd(A, full_matrices=False)


k = 50 #reduce k makes blurry

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