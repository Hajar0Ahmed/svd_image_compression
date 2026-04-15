import numpy as np
from scipy.optimize import brentq

# =============================================================================
# Algorithm 1: Golub-Reinsch SVD
# This implementation follows the "Two-Step" approach:
# 1. Bidiagonalization via Householder Reflections.
# 2. Iterative Diagonalization via Implicitly Shifted QR.

#Sources:
# https://www.cs.utexas.edu/~inderjit/public_papers/HLA_SVD.pdf
# https://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=9112096&fileOId=9112097
# =============================================================================

def algorithm_1a(A):
    '''
    STEP 1: Householder Bidiagonalization
    Reduces a general matrix to upper bidiagonal form.
    
    Parameters:
        A (ndarray): Input matrix of shape (m, n).
        
    Returns:
        B (ndarray): Upper bidiagonal matrix of shape (m, n).
        U (ndarray): m x m orthogonal matrix (Left transformations).
        V (ndarray): n x n orthogonal matrix (Right transformations).
        
    Assumptions:
        - The input matrix A is real-valued.
        - Per Golub-Reinsch requirements, we assume m >= n.
        
    Formulas:
        - Householder Vector: v = x + sign(x_1)||x||e_1
        - Householder Reflection: H = I - 2vv^T / (v^Tv)
        - Transformation: B' = H_m ... H_1 A P_1 ... P_{n-2}
    '''
    m, n = A.shape
    B = A.copy().astype(np.float64)
    U = np.identity(m)
    V = np.identity(n)

    for k in range(n):
        ## Left Transformation (Eliminate below B[k,k]) ##
        x = B[k:m, k]
        norm_x = np.linalg.norm(x)
        if norm_x > 1e-15:  # Slightly higher than machine epsilon
            s = np.sign(x[0] if x[0] != 0 else 1.0) * norm_x
            v = x.copy()
            v[0] += s
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-18:
                v /= v_norm
                # Apply to B and U
                B[k:m, k:] -= 2 * np.outer(v, v @ B[k:m, k:])
                U[:, k:m] -= 2 * (U[:, k:m] @ v)[:, None] @ v[None, :]

        ## Right Transformation (Eliminate right of B[k, k+1]) ##
        if k < n - 1:
            y = B[k, k+1:n]
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-30:
                y = np.zeros_like(y)
                y[0] = 1.0
                norm_y = 1.0

            s_row = np.sign(y[0]) * norm_y
            v_row = y.copy()
            v_row[0] += s_row

            v_row_norm = np.linalg.norm(v_row)
            if v_row_norm > 1e-30:
                v_row /= v_row_norm

                B[k:, k+1:n] -= 2 * (B[k:, k+1:n] @ v_row)[:, None] @ v_row[None, :]
                V[:, k+1:n] -= 2 * (V[:, k+1:n] @ v_row)[:, None] @ v_row[None, :]
                
    return B, U, V


def givens_coefficients(y, z):
    '''
    Computes rotation parameters to zero out the second component of a 2nd-order vector.
    
    Parameters:
        y (float): First component (to be preserved/modified).
        z (float): Second component (to be zeroed).
        
    Returns:
        c (float): Cosine of the rotation angle.
        s (float): Sine of the rotation angle.
        
    Formula:
        - r = sqrt(y^2 + z^2)
        - c = y/r, s = z/r
    '''
    if z == 0:
        return 1.0, 0.0
    
    if abs(z) > abs(y):
        tau = y / z
        s = 1.0 / np.sqrt(1 + tau**2)
        c = s * tau
    else:
        tau = z / y
        c = 1.0 / np.sqrt(1 + tau**2)
        s = c * tau
        
    return c, s


def algorithm_1c(B, U, V, p, q):
    '''
    STEP 2: Implicitly Shifted QR Step (Golub-Kahan SVD Step)
    Reduces the superdiagonal elements of the active submatrix.
    
    Parameters:
        B, U, V (ndarray): Current SVD matrices.
        p (int): Starting index of the active bidiagonal block.
        q (int): Number of already converged singular values at the end.
        
    Assumptions:
        - B is in upper bidiagonal form.
        - The submatrix B[p:n-q, p:n-q] is unreduced (no zeros on diagonal/superdiagonal).
        
    Formulas:
        - Wilkinson Shift (mu): Eigenvalue of the trailing 2x2 of B^T B closest to B_{nn}^2.
        - mu = d_n^2 + f_m^2 - (t12^2 / (delta + sign(delta) * sqrt(delta^2 + t12^2)))
    '''
    n = B.shape[0]
    i = n - q - 1 

    # Wilkinson Shift Calculation
    d_m, f_m, d_n = B[i-1, i-1], B[i-1, i], B[i, i]
    t11 = d_m**2 + f_m**2
    t12 = d_m * f_m
    t22 = d_n**2 + f_m**2
    delta = (t11 - t22) / 2.0
    sign_delta = np.sign(delta) if delta != 0 else 1.0
    mu = t22 - sign_delta * t12**2 / (abs(delta) + np.sqrt(delta**2 + t12**2))

    # Initial Implicit Shift
    y = B[p, p]**2 - mu
    z = B[p, p] * B[p, p+1]

    for k in range(p, n - q - 1):
        # Right Rotation G(k, k+1, theta) applied to V and B
        c, s = givens_coefficients(y, z)
        B[:, k], B[:, k+1] = c*B[:, k] + s*B[:, k+1], -s*B[:, k] + c*B[:, k+1]
        V[:, k], V[:, k+1] = c*V[:, k] + s*V[:, k+1], -s*V[:, k] + c*V[:, k+1]

        #Left Rotation G(k, k+1, phi) applied to U and B to restore bidiagonal form
        y, z = B[k, k], B[k+1, k]
        c, s = givens_coefficients(y, z)
        B[k, :], B[k+1, :] = c*B[k, :] + s*B[k+1, :], -s*B[k, :] + c*B[k+1, :]
        U[:, k], U[:, k+1] = c*U[:, k] + s*U[:, k+1], -s*U[:, k] + c*U[:, k+1]
        B[k+1, k] = 0.0 

        if k < n - q - 2:
            y, z = B[k, k+1], B[k, k+2]

    return B, U, V


def algorithm_1b(A, max_iter=1000, eps=1e-12):
    m, n = A.shape
    B, U, V = algorithm_1a(A)
    Bs = B[:n, :n].copy()

    for _ in range(max_iter):
        # Step 1: Zero small superdiagonal elements
        for i in range(n - 1):
            if abs(Bs[i, i+1]) <= eps * (abs(Bs[i, i]) + abs(Bs[i+1, i+1])):
                Bs[i, i+1] = 0.0

    
        q = 0
        for i in range(n - 2, -1, -1):
            if Bs[i, i+1] == 0.0:
                q += 1
            else:
                break
        if q == n - 1:
            break  

        p = 0
        for i in range(n - q - 2, -1, -1):
            if Bs[i, i+1] == 0.0:
                p = i + 1
                break

        # Step 3: Check for a zero diagonal in the active block
        zero_diag_idx = -1
        for i in range(p, n - q):
            if abs(Bs[i, i]) < eps:
                zero_diag_idx = i
                break

        if zero_diag_idx >= 0:
            i = zero_diag_idx

            if i < n - q - 1:
                # -------------------------------------------------------
                # Case A: Zero diagonal is NOT the last row of active block.
                # -------------------------------------------------------
                f = Bs[i, i+1]
                Bs[i, i+1] = 0.0

                for j in range(i + 1, n - q):
                    # Left rotation on rows i and j to zero f using Bs[j,j]
                    c, s = givens_coefficients(Bs[j, j], f)

                    new_jj = c * Bs[j, j] + s * f
                    f      = -s * Bs[j, j] + c * f
                    Bs[j, j] = new_jj

                    if j < n - q - 1:
                        f_next      = -s * Bs[j, j+1]
                        Bs[j, j+1] =  c * Bs[j, j+1]
                        f = f_next

                    # Left rotation updates U (acting on rows of B = cols of U^T)
                    U[:, i], U[:, j] = (
                         c * U[:, i] + s * U[:, j],
                        -s * U[:, i] + c * U[:, j]
                    )

            else:
                # -------------------------------------------------------
                # Case B: Zero diagonal IS the last row of the active block
                # (i == n-q-1). 
                # -------------------------------------------------------
                f = Bs[i - 1, i]
                Bs[i - 1, i] = 0.0

                # Single right rotation: zero Bs[i-1, i] using Bs[i-1, i-1]
                c, s = givens_coefficients(Bs[i - 1, i - 1], f)

                col_im1 = Bs[:, i - 1].copy()
                col_i   = Bs[:, i].copy()
                col_i[i - 1] = f  # restore the entry we just zeroed for the rotation

                Bs[:, i - 1] =  c * col_im1 + s * col_i
                Bs[:, i]     = -s * col_im1 + c * col_i

                # Right rotation updates V
                V[:, i - 1], V[:, i] = (
                     c * V[:, i - 1] + s * V[:, i],
                    -s * V[:, i - 1] + c * V[:, i]
                )

        else:
            Bs, U, V = algorithm_1c(Bs, U, V, p, q)

    sv_raw = np.diag(Bs)
    singular_values = np.abs(sv_raw)
    idx = np.argsort(singular_values)[::-1]

    S_diag = np.diag(singular_values[idx])
    V_sorted = V[:, idx]

    U_sorted = U.copy()
    U_sorted[:, :n] = U[:, idx]

    signs = np.where(sv_raw[idx] < 0, -1.0, 1.0)
    U_sorted[:, :n] *= signs[np.newaxis, :]

    return S_diag, U_sorted, V_sorted, singular_values[idx]



# =============================================================================
# Driver Function
# This ensures that assumptions are met for both algorithms
# =============================================================================

def svd_compressor_main(image_matrix):
    
    A_input = np.array(image_matrix, dtype=np.float64)
    m_orig, n_orig = A_input.shape
    
    is_transposed = False
    if m_orig < n_orig:
        A_input = A_input.T
        is_transposed = True

    m, n = A_input.shape 

    _, U_raw, V_raw, s_values = algorithm_1b(A_input)
    

    
    #Reassembly
    S_rect = np.zeros((m, n))
    k = len(s_values)
    S_rect[:k, :k] = np.diag(s_values)

    if is_transposed:
        return S_rect.T, V_raw, U_raw
    else:
        return S_rect, U_raw, V_raw