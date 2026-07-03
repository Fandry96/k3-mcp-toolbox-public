import numpy as np
import time

def bench(N, D):
    np.random.seed(42)
    matrix = np.random.randn(N, D).astype(np.float32)
    q_vec = np.random.randn(D).astype(np.float32)

    # Original
    t0 = time.time()
    for _ in range(100):
        m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        scores_orig = np.dot(m_norm, q_norm)
    t1 = time.time()

    # Optimized
    t2 = time.time()
    for _ in range(100):
        raw_scores = np.dot(matrix, q_vec)
        m_sq_norms = np.einsum('ij,ij->i', matrix, matrix)
        q_sq_norm = np.dot(q_vec, q_vec)
        scores_opt = raw_scores / ((np.sqrt(m_sq_norms) + 1e-9) * (np.sqrt(q_sq_norm) + 1e-9))
    t3 = time.time()

    print(f"N={N}, D={D}")
    print(f"Original: {(t1 - t0)*1000/100:.3f} ms")
    print(f"Optimized: {(t3 - t2)*1000/100:.3f} ms")
    print(f"Speedup: {(t1 - t0)/(t3 - t2):.2f}x")
    print(f"Max diff: {np.max(np.abs(scores_orig - scores_opt)):.2e}")

bench(10000, 64)
bench(1000, 768)
bench(100000, 64)
