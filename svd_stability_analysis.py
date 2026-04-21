# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 2026

@author: antwi
"""

import numpy as np
from Svd_algorithms import svd_compressor_main


def rel_err(a, b):
    na = np.linalg.norm(a)
    if na == 0:
        return np.linalg.norm(a - b)
    return np.linalg.norm(a - b) / na


def fro_err(a, b):
    na = np.linalg.norm(a, 'fro')
    if na == 0:
        return np.linalg.norm(a - b, 'fro')
    return np.linalg.norm(a - b, 'fro') / na


def two_err(a, b):
    na = np.linalg.norm(a, 2)
    if na == 0:
        return np.linalg.norm(a - b, 2)
    return np.linalg.norm(a - b, 2) / na


def get_s(S, p):
    s = np.diag(S)
    return s[:p]


def one_case(A, name):
    A = np.array(A, dtype=float)
    m, n = A.shape
    p = min(m, n)

    S, U, V = svd_compressor_main(A)

    A_rec = U @ S @ V.T

    s_custom = get_s(S, p)
    s_numpy = np.linalg.svd(A, compute_uv=False)

    Uh = U[:, :p]
    Vh = V[:, :p]
    Sh = np.diag(s_custom)

    rec_fro = fro_err(A, A_rec)
    rec_two = two_err(A, A_rec)

    orth_u = np.linalg.norm(U.T @ U - np.eye(U.shape[1]), 'fro')
    orth_v = np.linalg.norm(V.T @ V - np.eye(V.shape[1]), 'fro')

    sv_rel = rel_err(s_numpy, s_custom)
    sv_max = np.max(np.abs(s_custom - s_numpy))

    right_eq = np.linalg.norm(A @ Vh - Uh @ Sh, 'fro')
    left_eq = np.linalg.norm(A.T @ Uh - Vh @ Sh, 'fro')

    A_fro = np.linalg.norm(A, 'fro')
    if A_fro != 0:
        right_eq = right_eq / A_fro
        left_eq = left_eq / A_fro

    rank_custom = np.sum(s_custom > 1e-12)
    rank_numpy = np.linalg.matrix_rank(A, tol=1e-12)

    try:
        cond2 = np.linalg.cond(A)
    except:
        cond2 = np.inf

    out = []
    out.append(name)
    out.append(f'{m}x{n}')
    out.append(cond2)
    out.append(rec_fro)
    out.append(rec_two)
    out.append(orth_u)
    out.append(orth_v)
    out.append(sv_rel)
    out.append(sv_max)
    out.append(right_eq)
    out.append(left_eq)
    out.append(int(rank_custom))
    out.append(int(rank_numpy))

    return out


def hilbert(n):
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    return H


def build_cases():
    np.random.seed(0)

    cases = []

    cases.append(['square_random_8', np.random.randn(8, 8)])
    cases.append(['tall_random_12x6', np.random.randn(12, 6)])
    cases.append(['wide_random_6x12', np.random.randn(6, 12)])
    cases.append(['identity_8', np.eye(8)])
    cases.append(['zero_6', np.zeros((6, 6))])
    cases.append(['rank_one_8', np.outer(np.arange(1.0, 9.0), np.ones(8))])
    cases.append(['hilbert_8', hilbert(8)])
    cases.append(['ill_diag_6', np.diag([1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])])

    Q1, _ = np.linalg.qr(np.random.randn(10, 10))
    Q2, _ = np.linalg.qr(np.random.randn(10, 10))
    s = np.array([1.0, 1.0 - 1e-10, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16])
    A = Q1 @ np.diag(s) @ Q2.T
    cases.append(['clustered_sv_10', A])

    B = np.random.randn(8, 8)
    scales = [1e-12, 1e-8, 1e-4, 1.0, 1e4, 1e8]

    for a in scales:
        cases.append([f'scaled_random_8_alpha_{a:.0e}', a * B])

    return cases


def print_table(results):
    print('case'.ljust(28), '|',
          'shape'.ljust(6), '|',
          'cond2'.ljust(10), '|',
          'rec_fro'.ljust(10), '|',
          'orth_u'.ljust(10), '|',
          'orth_v'.ljust(10), '|',
          'sv_rel'.ljust(10), '|',
          'right_eq'.ljust(10), '|',
          'left_eq'.ljust(10))

    print('-' * 28, '+',
          '-' * 6, '+',
          '-' * 10, '+',
          '-' * 10, '+',
          '-' * 10, '+',
          '-' * 10, '+',
          '-' * 10, '+',
          '-' * 10, '+',
          '-' * 10)

    for r in results:
        name = r[0]
        shape = r[1]
        cond2 = r[2]
        rec_fro = r[3]
        orth_u = r[5]
        orth_v = r[6]
        sv_rel = r[7]
        right_eq = r[9]
        left_eq = r[10]

        if np.isfinite(cond2):
            cond2_txt = f'{cond2:.3e}'
        else:
            cond2_txt = 'inf'

        print(name.ljust(28), '|',
              shape.ljust(6), '|',
              cond2_txt.ljust(10), '|',
              f'{rec_fro:.3e}'.ljust(10), '|',
              f'{orth_u:.3e}'.ljust(10), '|',
              f'{orth_v:.3e}'.ljust(10), '|',
              f'{sv_rel:.3e}'.ljust(10), '|',
              f'{right_eq:.3e}'.ljust(10), '|',
              f'{left_eq:.3e}'.ljust(10))


def print_summary(results):
    recs = []
    orths = []
    svs = []

    for r in results:
        recs.append(r[3])
        orths.append(max(r[5], r[6]))
        svs.append(r[7])

    print('\nSummary:')
    print(f'best reconstruction error  : {min(recs):.3e}')
    print(f'worst reconstruction error : {max(recs):.3e}')
    print(f'worst orthogonality error  : {max(orths):.3e}')
    print(f'worst singular value error : {max(svs):.3e}')

    print('\nScaling diagnostic:')
    for r in results:
        if r[0].startswith('scaled_random_8_alpha_'):
            if np.isfinite(r[2]):
                cond2_txt = f'{r[2]:.3e}'
            else:
                cond2_txt = 'inf'

            print(f'{r[0]}: rec_fro={r[3]:.3e}, sv_rel={r[7]:.3e}, cond2={cond2_txt}')


def main():
    cases = build_cases()
    results = []

    for item in cases:
        name = item[0]
        A = item[1]
        out = one_case(A, name)
        results.append(out)

    print_table(results)
    print_summary(results)


if __name__ == '__main__':
    main()