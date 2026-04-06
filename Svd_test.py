import numpy as np
import pytest

# Import your functions from your implementation file
# Replace 'svd_implementation' with the actual name of your .py file
from Svd_algorithms import (
    algorithm_1a, algorithm_1b, algorithm_1c, 
    solve_equation, algorithm_5
)

class TestSVDSystem:
    """
    Comprehensive Test for SVD Algorithms
    """
    
    def setup_method(self):
        # Standard test matrix (3x3 symmetric)
        self.A_sym = np.array([
            [2, 1, 0],
            [1, 2, 1],
            [0, 1, 2]
        ], dtype=float)
        
        # Rectangular test matrix (4x3)
        self.A_rect = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ], dtype=float)

    # --- 1. Test Algorithm 1a (Bidiagonalization) ---
    def test_bidiagonalization(self):
        print("\nTesting 1a: Householder Bidiagonalization...")
        for A in [self.A_sym, self.A_rect]:
            m, n = A.shape
            B, U, V = algorithm_1a(m, n, A)
            
            # Identity: A = UBV^T
            reconstruction = U @ B @ V.T
            assert np.allclose(A, reconstruction, atol=1e-10)
            
            # Orthogonality: U'U = I, V'V = I
            assert np.allclose(U.T @ U, np.identity(m), atol=1e-10)
            assert np.allclose(V.T @ V, np.identity(n), atol=1e-10)
            
            # Structure: Elements below diag and above super-diag should be 0
            # Check lower triangle (excluding main diag)
            assert np.allclose(np.tril(B, -1), 0, atol=1e-12)
            # Check elements above first super-diagonal
            assert np.allclose(np.triu(B, 2), 0, atol=1e-12)

    # --- 2. Test Algorithm 1c (Implicit QR Step) ---
    def test_qr_step_invariants(self):
        print("\nTesting 1c: Singular Value Invariance in QR Step...")
        # Create a 3x3 bidiagonal matrix
        B = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 2.0]])
        n = 3
        
        expected_sv = np.sort(np.linalg.svd(B, compute_uv=False))
        
        # Run one step
        B_new, U_new, V_new = algorithm_1c(n, B.copy(), np.identity(3), np.identity(3), 0, 0)
        
        actual_sv = np.sort(np.linalg.svd(B_new, compute_uv=False))
        
        # The singular values must not change during the "chase"
        print(expected_sv)
        print(actual_sv)
        assert np.allclose(expected_sv, actual_sv, atol=1e-10)
        # Ensure it's still bidiagonal
        assert np.allclose(np.tril(B_new, -1), 0, atol=1e-12)

    # --- 3. Test Algorithm 1b (Golub-Reinsch Driver) ---
    def test_full_qr_svd(self):
        print("\nTesting 1b: Full Convergence and Accuracy...")
        A = self.A_sym
        m, n = A.shape
        
        S_comp, U, V = algorithm_1b(m, n, A)
        S_ref = np.sort(np.linalg.svd(A, compute_uv=False))[::-1]
        
        # Accuracy vs NumPy
        assert np.allclose(S_comp, S_ref, atol=1e-8)
        
        # Full Reconstruction A = USV^T
        Sigma = np.zeros((m, n))
        np.fill_diagonal(Sigma, S_comp)
        assert np.allclose(A, U @ Sigma @ V.T, atol=1e-8)

    # --- 4. Test Secular Solver (D&C) ---
    def test_secular_equation(self):
        print("\nTesting Secular Solver: Root Interlacing...")
        d = np.array([1.0, 2.0, 5.0])
        z = np.array([0.5, 0.5, 0.5])
        
        # Roots must interlace between d[i] and d[i+1]
        for i in range(len(d) - 1):
            root = solve_equation(d, z, i)
            assert d[i] < root < d[i+1]
            
            # Check if f(root) is actually zero
            f_val = 1.0 + np.sum(z**2 / (d**2 - root**2))
            assert abs(f_val) < 1e-10

    # --- 5. Test Algorithm 5 (Divide and Conquer) ---
    def test_divide_and_conquer(self):
        print("\nTesting Algorithm 5: Recursive SVD...")
        # Small bidiagonal matrix
        B = np.array([[5, 1], [0, 2]], dtype=float)
        n = 2
        
        # We check if D&C correctly handles the rank-one update
        # (Assuming your D&C logic for u_new/v_new is implemented)
        try:
            S_comp, U, V = algorithm_5(n, B, np.eye(3), np.eye(2))
            S_ref = np.sort(np.linalg.svd(B, compute_uv=False))[::-1]
            assert np.allclose(np.sort(S_comp)[::-1], S_ref, atol=1e-7)
        except Exception as e:
            pytest.fail(f"Algorithm 5 failed with error: {e}")

if __name__ == "__main__":
    # If running manually without pytest
    tester = TestSVDSystem()
    tester.setup_method()
    tester.test_bidiagonalization()
    tester.test_qr_step_invariants()
    tester.test_full_qr_svd()
    tester.test_secular_equation()
    print("\nAll individual algorithm tests completed successfully.")