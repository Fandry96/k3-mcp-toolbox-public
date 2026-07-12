import time
import numpy as np

np.random.seed(42)
N = 100000
D = 64
m_short = np.random.randn(N, D).astype(np.float32)
q_short = np.random.randn(D).astype(np.float32)

def orig():
    m_short_norm = m_short / (
        np.linalg.norm(m_short, axis=1, keepdims=True) + 1e-9
    )
    q_short_norm = q_short / (np.linalg.norm(q_short) + 1e-9)
    scores_short = np.dot(m_short_norm, q_short_norm)
    return scores_short

def optimized():
    m_sq_norms = np.einsum('ij,ij->i', m_short, m_short)
    q_sq_norm = np.dot(q_short, q_short)
    raw_scores = np.dot(m_short, q_short)
    denominator = (np.sqrt(m_sq_norms) + 1e-9) * (np.sqrt(q_sq_norm) + 1e-9)
    scores_short = raw_scores / denominator
    return scores_short

t0 = time.time()
for _ in range(100):
    s1 = orig()
t1 = time.time()
print(f"Original: {(t1-t0)/100 * 1000:.2f} ms")

t0 = time.time()
for _ in range(100):
    s2 = optimized()
t1 = time.time()
print(f"Optimized: {(t1-t0)/100 * 1000:.2f} ms")

print("Max diff:", np.max(np.abs(s1 - s2)))
