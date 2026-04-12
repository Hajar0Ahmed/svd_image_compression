import numpy as np

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
        ## Left Transformation ##
        x = B[k:m, k]
        norm_x = np.linalg.norm(x)
        if norm_x > 1e-15:
            s = np.sign(x[0] if x[0] != 0 else 1.0) * norm_x
            v = x.copy()
            v[0] += s
            v /= np.linalg.norm(v)
            
            # Apply to B and U
            B[k:m, k:] -= 2 * np.outer(v, v @ B[k:m, k:])
            U[:, k:m] -= 2 * (U[:, k:m] @ v)[:, None] @ v[None, :]

        ## Right Transformation ##
        # CHANGE: k < n - 1 and slice starting at k+1
        if k < n - 1:
            y = B[k, k+1:n]
            norm_y = np.linalg.norm(y)
            if norm_y > 1e-15:
                s_row = np.sign(y[0] if y[0] != 0 else 1.0) * norm_y
                v_row = y.copy()
                v_row[0] += s_row
                v_row /= np.linalg.norm(v_row)
                
                # Apply to B and V
                # We update the submatrix starting from column k+1
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
    r = np.hypot(y, z)
    if r == 0:
        return 1.0, 0.0
    return y/r, z/r


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
    '''
    STEP 3: The Iterative SVD Solver
    Governs the splitting of the matrix and convergence logic.
    
    Parameters:
        A (ndarray): m x n input matrix.
        max_iter (int): Safety limit to prevent infinite loops.
        eps (float): Machine precision tolerance for convergence.
        
    Returns:
        S (ndarray): Diagonal matrix of singular values.
        U (ndarray): Left singular vectors.
        V (ndarray): Right singular vectors.
        singular_values (ndarray): 1D array of sorted singular values.
    '''
    m, n = A.shape
    B, U, V = algorithm_1a(A)
    Bs = B[:n, :n].copy()
    U = U[:, :n].copy() 

    for _ in range(max_iter):
        # 1. Zeroing Small Superdiagonal Elements
        # Formula: |b_{i, i+1}| <= eps * (|b_{ii}| + |b_{i+1,i+1}|)
        for i in range(n - 1):
            if abs(Bs[i, i+1]) <= eps * (abs(Bs[i, i]) + abs(Bs[i+1, i+1])):
                Bs[i, i+1] = 0.0

        # 2. Identify Converged (q) and Active (p) Blocks
        q = 0
        for i in range(n - 2, -1, -1):
            if Bs[i, i+1] == 0.0: q += 1
            else: break
        if q == n - 1: break # Convergence achieved

        p = 0
        for i in range(n - q - 2, -1, -1):
            if Bs[i, i+1] == 0.0:
                p = i + 1
                break

        # 3. Apply Shifted QR or Handle Singular Diagonal Elements
        zero_diag_idx = -1
        for i in range(p, n - q):
            if abs(Bs[i, i]) < eps:
                zero_diag_idx = i
                break

        if zero_diag_idx >= 0:
            #zero out the super-diagonal element Bs[i, i+1]
            i = zero_diag_idx
            # If the zero is at Bs[i,i], we use Left rotations to zero Bs[i, i+1]
            if i < n - q - 1:
                f = Bs[i, i+1]
                Bs[i, i+1] = 0.0
                for j in range(i + 1, n - q):
                    c, s = givens_coefficients(Bs[j, j], f)
                    Bs[j, j] = np.hypot(Bs[j, j], f)
                    if j < n - q - 1:
                        f = -s * Bs[j, j+1]
                        Bs[j, j+1] = c * Bs[j, j+1]
                    
                    # Update U (Left rotations)
                    U[:, i], U[:, j] = c*U[:, i] - s*U[:, j], s*U[:, i] + c*U[:, j]
        else:
            Bs, U, V = algorithm_1c(Bs, U, V, p, q)

    #Ensure non-negativity and descending order
    sv_raw = np.diag(Bs)
    signs = np.where(sv_raw < 0, -1.0, 1.0)
    U = U * signs[np.newaxis, :]
    singular_values = np.abs(sv_raw)

    idx = np.argsort(singular_values)[::-1]
    return np.diag(singular_values[idx]), U[:, idx], V[:, idx], singular_values[idx]


def svd_compressor_main(image_matrix):
    '''
    Driver for SVD-based Image Compression.
    
    Parameters:
        image_matrix (ndarray): Input image as a 2D float array.
        
    Returns:
        S, U, V (ndarray): The full SVD decomposition components.
        
    Assumptions:
        - Handles both tall (m > n) and wide (m < n) matrices by using 
          the property: A = U S V^T  => A^T = V S^T U^T.
    '''
    A_input = np.array(image_matrix, dtype=np.float64)
    m, n = A_input.shape
    
    is_transposed = False
    if m < n:
        # If wide, transpose to satisfy m >= n for the algorithm.
        A_input = A_input.T
        is_transposed = True

    # curr_m is now always >= curr_n
    curr_m, curr_n = A_input.shape
    S_bidiag, U_raw, V_raw, s_values = algorithm_1b(A_input)

    # For Thin SVD: S is square (k, k), U is (m, k), V is (n, k).
    # Here k = curr_n.
    S_working = np.diag(s_values)

    if is_transposed:
        # If we transposed: Original A was (m, n).
        # Algorithm ran on A.T (n, m).
        # U_raw is (n, m), V_raw is (m, m).
        # A = (V_raw) @ (S_working.T) @ (U_raw.T)
        return S_working.T, V_raw, U_raw
    else:
        # A = U_raw @ S_working @ V_raw.T
        return S_working, U_raw, V_raw