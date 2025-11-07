import numpy as np

def qr_decomposition(matrix):
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Parameters:
    matrix (np.ndarray): Input matrix A
    
    Returns:
    tuple: Q (orthogonal matrix), R (upper triangular matrix)
    """
    A = matrix.copy().astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        # Start with the j-th column of A
        v = A[:, j].copy()
        
        # Subtract projections onto previous columns
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        # Normalize to get the j-th column of Q
        R[j, j] = np.linalg.norm(v)
        if R[j, j] != 0:
            Q[:, j] = v / R[j, j]
    
    return Q, R

def qr_eigenvalue_approximation(matrix, iterations=10):
    """
    Approximate eigenvalues using the QR algorithm.
    
    Parameters:
    matrix (np.ndarray): Input square matrix
    iterations (int): Number of QR iterations to perform
    
    Returns:
    tuple: (final_matrix, eigenvalue_approximations)
    """
    A = matrix.copy().astype(float)
    n = A.shape[0]
    
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
    
    # Compare with numpy's eigenvalue function
    true_eigenvalues = np.linalg.eigvals(matrix)
    print(f"True eigenvalues (numpy):  {np.sort(true_eigenvalues)}")
    print(f"Our approximation:         {np.sort(eigenvalues)}")
    print(f"Error: {np.sort(np.abs(np.sort(true_eigenvalues) - np.sort(eigenvalues)))}")