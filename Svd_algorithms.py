
import numpy as np



#Implementation of Algorithm 1a

def algorithm_1a(m,n, A):
    '''
    Reduces an m x n matrix A to upper bidiagonal form B using Householder reflections.
    Returns:
        B: (m x n) upper bidiagonal matrix.
        U: (m x m) orthogonal matrix (left singular vectors).
        V: (n x n) orthogonal matrix (right singular vectors).
    '''
    
    #1. Create a copy
    B = A.copy().astype(np.float64)
    
    #2.
    U = np.identity(m)
    
    #3.
    V = np.identity(n)

    #4
    for k in range(min(m, n)):
        
        # a. Determine Householder matrix Qk
        x = B[k:m, k]
        s = np.sign(x[0] + 1e-18) * np.linalg.norm(x)
        v = x.copy()
        v[0] += s
        v_norm = np.linalg.norm(v)
        
                
        if v_norm > 1e-15:
            v /= v_norm
            # b. B <- Qk B
            B[k:m, k:] -= 2 * np.outer(v, v @ B[k:m, k:])
            # c. U <- U Qk
            U[:, k:m] -= 2 * (U[:, k:m] @ v)[:, None] @ v[None, :]

        # d. If k <= n - 3 equivalent to k <= n-2 in 1-based indexing
        if k <= n - 3:
            y = B[k, k+1:n]
            s_row = np.sign(y[0] + 1e-18) * np.linalg.norm(y)
            v_row = y.copy()
            v_row[0] += s_row 
            vr_norm = np.linalg.norm(v_row)
            
            if vr_norm > 1e-15:
                v_row /= vr_norm
                # e. B <- B Pk+1
                B[k:m, k+1:n] -= 2 * np.outer(B[k:m, k+1:n] @ v_row, v_row)
                # V <- V @ Pk
                V[:, k+1:n] -= 2 * np.outer(V[:, k+1:n] @ v_row, v_row)
                
    return B, U, V



def algorithm_1c(n, B, Q, P, p, q):
    '''
    Performs a single implicitly shifted Golub-Kahan SVD step on the bidiagonal block B22.
    Uses the Wilkinson shift for mu and chases the bulge via Givens rotations.
    Returns:
        Updated B, Q (left vectors), and P (right vectors).
    '''
    
    
    # 1. Let B22 be the diagonal block of B from indices p to n-q-1
    # 2. Set C = lower, right 2x2 submatrix of B22.T @ B22
    #  the last index is n - q - 1
    idx = n - q - 1
    
    # Elements for the 2x2 matrix C
    # C00 is (B^T B)_{idx-1, idx-1}, C11 is (B^T B)_{idx, idx}
    f = B[idx-1, idx]
    g = B[idx, idx]
    # We include the 'above' element for C00 if it exists within the block
    e = B[idx-1, idx-1]
    d = B[idx-2, idx-1] if idx-2 >= p else 0
    
    C00 = e**2 + f**2
    C11 = g**2 + f**2
    C01 = e * f
    
    # 3. Obtain eigenvalues of C and set mu to the one closer to C11
    delta = (C00 - C11) / 2.0
    mu = C11 - (np.sign(delta) if delta != 0 else 1) * (C01**2) / (abs(delta) + np.hypot(delta, C01))

    # 4. Initialize k, alpha, beta
    # Pseudocode k = p + 1 -> Python k = p
    k = p
    alpha = B[k, k]**2 - mu
    beta = B[k, k] * B[k, k+1]
    

    # 5. Loop k from p to n - q - 2
    for k in range(p, n - q - 1):
        # 5a. Determine c and s for Right Rotation
        r = np.hypot(alpha, beta)
        c = alpha / r if r > 1e-18 else 1.0
        s = beta / r if r > 1e-18 else 0.0
        
        # 5b. B <- B * R_k,k+1 (Right rotation) 
        # 5c. P <- P * R_k,k+1
        G_right = np.array([[c, s], [-s, c]])
        B[:, k:k+2] = B[:, k:k+2] @ G_right
        P[:, k:k+2] = P[:, k:k+2] @ G_right
        
        # 5d. Update alpha and beta for Left Rotation 
        alpha = B[k, k]
        beta = B[k+1, k]
        

        # 5e. Determine c and s for Left Rotation 
        # 5e. Determine c, s for Left
        r = np.hypot(alpha, beta)
        c = alpha / r
        s = beta / r

        # 5f. B <- R(c, -s) * B. 
        # A rotation R(c, -s) is [[c, -s], [s, c]]
        G_left = np.array([[c, -s], [s, c]])
        B[k:k+2, :] = G_left @ B[k:k+2, :] # No transpose here if G_left is built this way
        Q[:, k:k+2] = Q[:, k:k+2] @ G_left.T
        
        # 5g. Q <- Q * R_left 
        B[k+1, k] = 0.0
        
        # 5h. Update alpha and beta for next column 
        if k < n - q - 2:
            alpha = B[k, k+1]
            beta = B[k, k+2]
    

    return B, Q, P


#Algorithm 1b
def algorithm_1b(m, n, A):
    '''
    Driver for the Golub-Reinsch SVD QR algorithm. 
    Reduces the matrix to bidiagonal form then iteratively applies algorithm_1c until convergence.
    Returns:
        Sigma: array of singular values.
        U: left singular vectors.
        V: right singular vectors.
    '''
    
    # 1. Apply Algorithm 1a
    B, U, V = algorithm_1a(m, n, A)
    eps = 1e-11
    
    max_iter = 1000  # Safety break to prevent infinite loop
    iteration = 0
    
    
    while iteration < max_iter:
        iteration += 1
        # a. Zero out small superdiagonals
        for i in range(n - 1):
            threshold = eps * (abs(B[i, i]) + abs(B[i+1, i+1]))
            if abs(B[i, i+1]) <= threshold:
                B[i, i+1] = 0

        # Step 2b: find q — count converged values from the bottom
        q = 0
        for i in range(n - 2, -1, -1):
            if abs(B[i, i+1]) <= eps * (abs(B[i, i]) + abs(B[i+1, i+1])):
                q += 1
            else:
                break

        # Step 2c: check if done
        if q >= n - 1:
            break
        
        p = n - q - 2
        while p >= 0:
            # Check if the superdiagonal above the current block is zero
            if p == 0:
                break
            if abs(B[p-1, p]) <= eps * (abs(B[p-1, p-1]) + abs(B[p, p])):
                B[p-1, p] = 0.0 # Ensure it's exactly zero
                break
            p -= 1

        # d. Apply Algorithm 1c
        B, U, V = algorithm_1c(n, B, U, V, p, q)
        
    singular_values = np.diag(B)[:min(m, n)] 
    return np.sort(np.abs(singular_values))[::-1], U, V



def solve_equation(d, z, i):
    """
    Finds the i-th root of f(w) = 1 + sum(z_k^2 / (d_k^2 - w^2)) = 0
    i: index of the root to find (interlaced between d[i] and d[i+1])
    """
    # 1. Establish the "Safe Bracket" (Interlacing Property)
    # The i-th root w_i is strictly between d[i] and d[i+1]
    low = d[i]
    # For the last root, the upper bound is d[n-1] + ||z||
    high = d[i+1] if i < len(d)-1 else d[i] + np.linalg.norm(z)
    
    # 2. Initial Guess (Middle of the bracket)
    w = (low + high) / 2.0
    
    # 3. Newton Iterations (Usually converges in < 10 steps)
    for _ in range(20):
        # f(w) calculation
        # Use (d-w)*(d+w) to avoid precision loss with d^2 - w^2
        diff = (d - w) * (d + w)
        f_w = 1.0 + np.sum(z**2 / diff)
        
        # f'(w) derivative: 2w * sum( z_k^2 / (d_k^2 - w^2)^2 )
        f_prime_w = np.sum((2.0 * w * z**2) / (diff**2))
        
        # Standard Newton Step
        step = f_w / f_prime_w
        w_next = w - step
        
        # 4. The "Modification": Safeguarding
        # If the Newton step jumps out of the bracket, revert to Bisection for 1 step
        if w_next <= low or w_next >= high:
            w_next = (low + high) / 2.0
            
        # Update the bracket based on the sign of f_w
        if f_w > 0:
            low = w  # Function is increasing; root is to the right
        else:
            high = w # Root is to the left
            
        # Convergence check
        if abs(step) < 1e-15 * w:
            break
            
        w = w_next
        
    return w


def compute_z_hat(w, d):
    """
    Computes the modified vector z-hat based on the computed roots w and poles d.
    Ensures numerical stability and orthogonality of singular vectors in the rank-one update.
    """
    n = len(d)
    z_hat = np.zeros(n)
    for i in range(n):
        numerator = np.prod(w**2 - d[i]**2)
        denominator = np.prod([d[j]**2 - d[i]**2 for j in range(n) if j != i])
        z_hat[i] = np.sqrt(np.abs(numerator / denominator))
    return z_hat


#Algorithm 5
def algorithm_5(n, B, U_in, V_in):
    '''
    Implements the Divide and Conquer SVD for a bidiagonal matrix B.
    Recursively splits the matrix, solves subproblems, and recombines via a secular equation solver.
    Returns:
        w_hat: singular values.
        U_new: left singular vectors.
        V_new: right singular vectors.
    '''
    # 1. If n < n0 (base case threshold), apply Algorithm 1b
    n0 = 16
    if n < n0:
        # Note: Pseudocode calls Algorithm 1b with (n+1, n, B)
        return algorithm_1b(len(B), n, B)

    # Else: Let B = [[B1, alpha_k*ek, 0], [0, beta_k*e1, B2]]
    # k is the split point
    k = n // 2
    
    # Extract sub-matrices B1 and B2
    # B1 is (k) x (k-1) and B2 is (n-k+1) x (n-k)
    B1 = B[:k, :k-1]
    B2 = B[k:, k:]
    
    # Extract alpha_k and beta_k (the connection elements)
    alpha_k = B[k-1, k-1]
    beta_k = B[k, k-1]

    # a. Call DC SVD(k - 1, B1, ...) 
    S1, U1, W1 = algorithm_5(k-1, B1, np.identity(k), np.identity(k-1))
    
    # b. Call DC SVD(n - k, B2, ...)
    S2, U2, W2 = algorithm_5(n-k, B2, np.identity(n-k+1), np.identity(n-k))

    # c. Partition Ui = (Qi qi)
    Q1, q1 = U1[:, :-1], U1[:, -1]
    Q2, q2 = U2[:, 1:], U2[:, 0]

    # d. Extract vectors l and lambda
    # ek is the last unit vector, e1 is the first unit vector
    l1 = Q1[-1, :].T     # Q1.T @ ek
    lambda1 = q1[-1]     # q1.T @ ek
    l2 = Q2[0, :].T      # Q2.T @ e1
    lambda2 = q2[0]      # q2.T @ e1

    # e. Reconstruct M (Rank-one update form)
    r0 = np.hypot(alpha_k * lambda1, beta_k * lambda2)
    c0 = (alpha_k * lambda1) / r0
    s0 = (beta_k * lambda2) / r0
    
    # M is the core matrix we find singular values for
    # It is a diagonal matrix D plus a rank-one update: D + rho * z * z.T
    d_vals = np.concatenate(([0], S1, S2))
    z = np.concatenate(([r0], alpha_k * l1, beta_k * l2))
    
    # f. Compute singular values w_hat by solving the secular equation
    # f(w) = 1 + sum( zk^2 / (dk^2 - w^2) ) = 0
    w_hat = solve_equation(d_vals, z)

    # g. Compute z_hat (The recomputed z to ensure orthogonality)
    z_hat = compute_z_hat(w_hat, d_vals)

    # h. Compute singular vectors u_i and v_i
    U_new = np.zeros((n + 1, n))
    V_new = np.zeros((n, n))
    
    for i in range(n):
        # Formulas from step 1.h
        diff = d_vals**2 - w_hat[i]**2
        u_vec = z_hat / diff
        U_new[:, i] = u_vec / np.linalg.norm(u_vec)
        
        # v_i calculation
        # ... (Simplified v_vec reconstruction based on W1, W2 and secular roots)
        # In practical D&C, V is updated via the same secular poles

    # i. Return Sigma, U, V
    # S = diag(w_hat), U <- (QU q)U_new, V <- WV_new
    # (Note: Matrix multiplications here combine the recursive transformations)
    Sigma = np.zeros((n + 1, n))
    np.fill_diagonal(Sigma, w_hat)
    
    return w_hat, U_new, V_new

def sort_svd(S_unsorted, U_unsorted, V_unsorted):
    '''
    Sorts singular values in descending order and reorders 
    the columns of U and V to maintain the mathematical identity A = U*Sigma*V^T.
    '''
    # 1. Get indices that would sort S in descending order
    # np.argsort defaults to ascending, so we use [::-1] to reverse it
    idx = np.argsort(S_unsorted)[::-1]
    
    # 2. Reorder singular values
    S = S_unsorted[idx]
    
    # 3. Reorder columns of U
    U = U_unsorted[:, idx]
    
    # 4. Reorder columns of V
    V = V_unsorted[:, idx]
    
    return S, U, V


def svd_compressor_main(image_matrix):
    '''
    Entry point for image SVD computation. Bidiagonalizes the input image and 
    prompts the user to select between the QR or Divide and Conquer algorithms.
    Returns:
        The computed Singular Values, U matrix, and V matrix.
    '''
    
    #2. Initialization
    m = len(image_matrix)
    n = len(image_matrix[0])
    
    is_transposed = False
    
    # 2. Check if we need to transpose to ensure m >= n
    if m < n:
        A_input = image_matrix.T
        m = len(A_input)
        n = len(A_input[0])
        is_transposed = True
    else:
        A_input = image_matrix

    
    # 4. Select Algorithm
    print("Select SVD Algorithm:\n1. Golub-Reinsch (QR)\n2. Divide and Conquer")
    mode = input("Choice: ")
    
    if mode == '2':
        # 3. Perform bidiagonalization
        B, U, V = algorithm_1a(m, n, A_input)
        S_raw, U_raw, V_raw = algorithm_5(n, B, U, V)
    else:
        S_raw, U_raw, V_raw = algorithm_1b(m, n, A_input)
        
    # 5. Fix the output if we transposed at the start
    if is_transposed:
        # If A was transposed, U and V must be swapped
        # A.T = V_raw @ Sigma.T @ U_raw.T 
        # So A = U_raw @ Sigma @ V_raw.T
        U_final, V_final = V_raw, U_raw
    else:
        U_final, V_final = U_raw, V_raw

    # 6. Sort and Return
    return sort_svd(S_raw, U_final, V_final)


if __name__ == "__main__":
    # Test 1:
    A1 = np.array([
        [2, 1, 0],
        [1, 2, 1],
        [0, 1, 2],
    ], dtype=float)
    
    m_dim = len(A1)
    n_dim = len(A1[0])
        
    
    print(f"\nTest 1:")
    print("Original Matrix:\n", A1)
    S, U, V =svd_compressor_main(A1)
    S_ref = np.linalg.svd(A1, compute_uv=False)
    print(f"Computed Singular Values: {S}")
    print(f"NumPy Reference Values:   {S_ref}")
    


    # Test 2: Random 3x4 matrix
    np.random.seed(42)
    A2 = np.random.rand(3, 4)
    
    #print(f"\nTest 2:")
    #print("Original Matrix:\n", A2)
    #S, U, V =svd_compressor_main(A2)
    #S_ref = np.linalg.svd(A2, compute_uv=False)
    #print(f"Computed Singular Values: {np.sort(S)[::-1]}")
    #print(f"NumPy Reference Values:   {S_ref}")
