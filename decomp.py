import numpy as np

def qr_decomposition(A):
    A = A.copy().astype(float)
    m, n = A.shape
    Q = np.zeros((m, n)) # initialize Q with zeroes
    Q = gram_schmidt(A)   # use gram schmidt to find Q
    R = np.zeros((n, n)) # initialize R with zeroes
    R = find_upper_triangular(Q, A) # R = Q^T * A

    return Q, R

def gram_schmidt(A):
    A = np.copy(A).astype(float) 
    n = A.shape[1]
    for j in range(n):
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        if np.isclose(np.linalg.norm(A[:, j]), 0, rtol=1e-15, atol=1e-14, equal_nan=False):
            A[:, j] = np.zeros(A.shape[0])
        else:    
            A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A

def find_upper_triangular(Q, A):
    return np.dot(Q.T, A)

def qr_eigenvalue_approximation(matrix, iterations=10):
    A = matrix.copy().astype(float)
    
    print(f"Starting QR algorithm with {iterations} iterations...")
    print(f"Initial matrix:\n{A}\n")
    
    for i in range(iterations):
        Q, R = qr_decomposition(A)
        A = np.dot(R, Q)  # A_new = R * Q
        
        print(f"Iteration {i+1}:")
        print(f"Matrix A:\n{A}")
        print(f"Diagonal (eigenvalue approximations): {np.diag(A)}\n")
    
    eigenvalues = np.diag(A)
    return A, eigenvalues

if __name__ == "__main__":
    # Example usage
    matrix = np.array([[4, 1, 2, 0, 0],
                       [1, 3, 0, 1, 0],
                       [2, 0, 5, 0 , 1],
                       [0, 1, 0, 2, 1],
                       [0, 0, 1, 1, 4]], dtype=float)

    print("Original Matrix:")
    print(matrix)
    print()
    
    # Perform QR algorithm to approximate eigenvalues
    final_matrix, eigenvalues = qr_eigenvalue_approximation(matrix, iterations=10)
    
    print("="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print(f"Final matrix after 10 QR iterations:")
    print(final_matrix)
    print(f"\nApproximated eigenvalues: {eigenvalues}")
