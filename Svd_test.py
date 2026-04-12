import numpy as np
import pytest

# Ensure the filename 'Svd_algorithms' matches your actual .py file
from Svd_algorithms import svd_compressor_main

def assert_svd_properties(A, S, U, V, tol=1e-9):
    '''
    Helper to verify the fundamental properties of SVD.
    
    Parameters:
        A (ndarray): Original input matrix.
        S (ndarray): Computed diagonal singular value matrix.
        U (ndarray): Computed left singular vectors.
        V (ndarray): Computed right singular vectors.
        tol (float): Numerical tolerance for comparisons.
    '''
    m, n = A.shape
    
    # PROPERTY 1: Reconstruction (A ≈ U @ S @ V.T)
    A_reconstructed = U @ S @ V.T
    np.testing.assert_allclose(A, A_reconstructed, atol=tol, 
                               err_msg="Reconstruction A = USV^T failed")

    # PROPERTY 2: Orthogonality of U (Columns are orthonormal)
    # U should be (m, k). U.T @ U should be (k, k) identity.
    k = S.shape[0] if m >= n else S.shape[1]
    expected_u_identity = np.eye(U.shape[1])
    np.testing.assert_allclose(U.T @ U, expected_u_identity, atol=tol, 
                               err_msg="U is not semi-orthogonal")

    # PROPERTY 3: Orthogonality of V (Columns are orthonormal)
    # V should be (n, k). V.T @ V should be (k, k) identity.
    expected_v_identity = np.eye(V.shape[1])
    np.testing.assert_allclose(V.T @ V, expected_v_identity, atol=tol, 
                               err_msg="V is not semi-orthogonal")

    # PROPERTY 4: Singular Values
    s = np.diag(S) if S.ndim > 1 else S
    assert np.all(s >= -tol), "Found negative singular values"
    
    # Check 4b: Sorted order (np.diff <= 0 for descending)
    diffs = np.diff(s)
    assert np.all(diffs <= 1e-12), f"Singular values not sorted: {s}"


### --- TEST CASES ---

def test_square_matrix():
    '''CASE: Basic 3x3 square matrix with known values.'''
    A = np.array([[4, 11, 14], [8, 7, -2], [1, -5, 3]], dtype=float)
    S, U, V = svd_compressor_main(A)
    assert_svd_properties(A, S, U, V)

def test_tall_matrix():
    '''CASE: m > n (more rows than columns).'''
    A = np.random.randn(10, 5)
    S, U, V = svd_compressor_main(A)
    assert_svd_properties(A, S, U, V)

def test_wide_matrix():
    '''CASE: m < n (more columns than rows).'''
    A = np.random.randn(5, 10)
    S, U, V = svd_compressor_main(A)
    assert_svd_properties(A, S, U, V)

def test_singular_matrix():
    '''CASE: Matrix with rank 1 (linearly dependent rows/cols).'''
    A = np.outer(np.array([1, 2, 3]), np.array([1, 1, 1]))
    S, U, V = svd_compressor_main(A)
    assert_svd_properties(A, S, U, V)
    
    # Verify only the first singular value is significant
    s = np.diag(S)
    assert s[0] > 1e-10
    np.testing.assert_allclose(s[1:], 0, atol=1e-10)

def test_zero_matrix():
    '''CASE: All elements are zero.'''
    A = np.zeros((4, 4))
    S, U, V = svd_compressor_main(A)
    assert_svd_properties(A, S, U, V)
    assert np.all(np.diag(S) <= 1e-15)

def test_identity_matrix():
    '''CASE: Identity matrix (Singular values should all be 1).'''
    A = np.eye(5)
    S, U, V = svd_compressor_main(A)
    assert_svd_properties(A, S, U, V)
    np.testing.assert_allclose(np.diag(S), np.ones(5), atol=1e-10)

def test_comparison_with_numpy():
    '''GOAL: Compare result against NumPy's library standard.'''
    A = np.random.rand(8, 5)
    S, _, _ = svd_compressor_main(A)
    expected_s = np.linalg.svd(A, compute_uv=False)
    
    # Verify S-diagonal matches np.linalg.svd
    np.testing.assert_allclose(np.diag(S)[:len(expected_s)], expected_s, atol=1e-9)