import time
import numpy as np

# Create a mock matrix
N = 100_000
D = 64
matrix = np.random.rand(N, D).astype(np.float32)

t0 = time.perf_counter()
norm1 = np.linalg.norm(matrix, axis=1, keepdims=True)
t1 = time.perf_counter()
print(f"np.linalg.norm (axis=1): {t1-t0:.5f}s")

t0 = time.perf_counter()
norm2 = np.sqrt(np.einsum('ij,ij->i', matrix, matrix))[:, np.newaxis]
t1 = time.perf_counter()
print(f"np.einsum: {t1-t0:.5f}s")

# Create a single vector
q = np.random.rand(D).astype(np.float32)

t0 = time.perf_counter()
norm3 = np.linalg.norm(q)
t1 = time.perf_counter()
print(f"np.linalg.norm (1d): {t1-t0:.6f}s")

t0 = time.perf_counter()
norm4 = np.sqrt(np.dot(q, q))
t1 = time.perf_counter()
print(f"np.dot: {t1-t0:.6f}s")
