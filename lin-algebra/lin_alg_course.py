"""
LINEAR ALGEBRA COURSE: From Basics to Eigenvectors
A pedagogical approach with Python implementations
"""

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

print("=" * 70)
print("LINEAR ALGEBRA COURSE: EIGENVECTORS & EIGENVALUES")
print("=" * 70)

# ============================================================================
# LESSON 1: Vectors - The Foundation
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 1: VECTORS - Building Blocks")
print("=" * 70)

# What is a vector?
v1 = np.array([3, 2])
v2 = np.array([1, 4])

print("\nVector v1:", v1)
print("Vector v2:", v2)

# Vector operations
print("\nVector addition: v1 + v2 =", v1 + v2)
print("Scalar multiplication: 2 * v1 =", 2 * v1)
print("Dot product: v1 · v2 =", np.dot(v1, v2))
print("Magnitude of v1:", np.linalg.norm(v1))

# ============================================================================
# LESSON 2: Matrices - Linear Transformations
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 2: MATRICES - Transforming Space")
print("=" * 70)

# A matrix transforms vectors
A = np.array([[2, 1],
              [1, 2]])

print("\nMatrix A:")
print(A)

v = np.array([1, 0])
print("\nOriginal vector v:", v)
print("Transformed vector A·v:", A @ v)

# Matrix multiplication
B = np.array([[1, 3],
              [2, 1]])
print("\nMatrix B:")
print(B)
print("\nMatrix multiplication A·B:")
print(A @ B)

# ============================================================================
# LESSON 3: Understanding Linear Transformations Visually
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 3: VISUALIZING TRANSFORMATIONS")
print("=" * 70)

def plot_transformation(A, title="Linear Transformation"):
    """Visualize how a matrix transforms the unit vectors"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original space
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(-1, 4)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Unit vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    ax1.quiver(0, 0, e1[0], e1[1], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.01, label='e₁ = [1,0]')
    ax1.quiver(0, 0, e2[0], e2[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.01, label='e₂ = [0,1]')
    ax1.set_title('Original Space')
    ax1.legend()
    
    # Transformed space
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-1, 4)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    
    # Transformed unit vectors
    Ae1 = A @ e1
    Ae2 = A @ e2
    
    ax2.quiver(0, 0, Ae1[0], Ae1[1], angles='xy', scale_units='xy', scale=1, 
               color='red', width=0.01, label=f'A·e₁ = {Ae1}')
    ax2.quiver(0, 0, Ae2[0], Ae2[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', width=0.01, label=f'A·e₂ = {Ae2}')
    ax2.set_title('Transformed Space')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

A_example = np.array([[2, 1],
                      [1, 2]])
print("\nVisualizing transformation by matrix A:")
print(A_example)
plot_transformation(A_example, "How Matrix A Transforms Space")

# ============================================================================
# LESSON 4: Special Directions - Introduction to Eigenvectors
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 4: THE MAGIC OF EIGENVECTORS")
print("=" * 70)

print("\nKey Insight: Some vectors only get SCALED, not rotated!")
print("These special vectors are called EIGENVECTORS")
print("The scaling factor is called the EIGENVALUE")

A = np.array([[4, 2],
              [1, 3]])

print("\nMatrix A:")
print(A)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nEigenvalues (λ):", eigenvalues)
print("\nEigenvectors:")
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    print(f"\nEigenvector {i+1}: {vec}")
    print(f"Eigenvalue {i+1}: {val}")
    
    # Verify: A·v = λ·v
    Av = A @ vec
    lambda_v = val * vec
    print(f"A·v = {Av}")
    print(f"λ·v = {lambda_v}")
    print(f"Are they equal? {np.allclose(Av, lambda_v)}")

# ============================================================================
# LESSON 5: Visualizing Eigenvectors
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 5: SEEING EIGENVECTORS IN ACTION")
print("=" * 70)

def plot_eigenvectors(A):
    """Visualize eigenvectors and their transformations"""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    colors = ['red', 'blue']
    
    for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        # Original eigenvector
        ax.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
                 color=colors[i], width=0.008, alpha=0.5, 
                 label=f'v{i+1} (eigenvector)')
        
        # Transformed eigenvector
        Av = A @ vec
        ax.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1,
                 color=colors[i], width=0.012, 
                 label=f'A·v{i+1} = {val:.2f}·v{i+1}')
    
    # Regular vector for comparison
    regular_v = np.array([1, 1])
    Av_regular = A @ regular_v
    ax.quiver(0, 0, regular_v[0], regular_v[1], angles='xy', scale_units='xy', 
             scale=1, color='green', width=0.006, alpha=0.5, label='regular vector')
    ax.quiver(0, 0, Av_regular[0], Av_regular[1], angles='xy', scale_units='xy',
             scale=1, color='green', width=0.01, label='transformed (rotated!)')
    
    ax.set_title('Eigenvectors Stay on Their Line!\nRegular vectors rotate')
    ax.legend()
    plt.tight_layout()
    plt.show()

print("\nVisualizing eigenvectors:")
plot_eigenvectors(A)

# ============================================================================
# LESSON 6: Computing Eigenvalues - The Characteristic Equation
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 6: THE MATH BEHIND EIGENVALUES")
print("=" * 70)

print("\nTo find eigenvalues, we solve: det(A - λI) = 0")
print("This is called the characteristic equation")

A = np.array([[3, 1],
              [1, 3]])

print("\nMatrix A:")
print(A)

# Manual calculation for 2x2 matrix
print("\nFor a 2x2 matrix [[a,b],[c,d]]:")
print("Characteristic equation: λ² - (a+d)λ + (ad-bc) = 0")

a, b = A[0, 0], A[0, 1]
c, d = A[1, 0], A[1, 1]

trace = a + d
det = a*d - b*c

print(f"\nTrace (a+d) = {trace}")
print(f"Determinant (ad-bc) = {det}")
print(f"Characteristic equation: λ² - {trace}λ + {det} = 0")

# Solve using quadratic formula
discriminant = trace**2 - 4*det
lambda1 = (trace + np.sqrt(discriminant)) / 2
lambda2 = (trace - np.sqrt(discriminant)) / 2

print(f"\nEigenvalues: λ₁ = {lambda1:.4f}, λ₂ = {lambda2:.4f}")
print(f"NumPy verification: {np.linalg.eigvals(A)}")

# ============================================================================
# LESSON 7: Properties of Eigenvalues and Eigenvectors
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 7: IMPORTANT PROPERTIES")
print("=" * 70)

A = np.array([[2, 1, 0],
              [1, 2, 1],
              [0, 1, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nMatrix A:")
print(A)

print("\n1. Sum of eigenvalues = Trace of matrix")
print(f"   Sum of eigenvalues: {np.sum(eigenvalues):.6f}")
print(f"   Trace of A: {np.trace(A):.6f}")

print("\n2. Product of eigenvalues = Determinant of matrix")
print(f"   Product of eigenvalues: {np.prod(eigenvalues):.6f}")
print(f"   Determinant of A: {np.linalg.det(A):.6f}")

print("\n3. Eigenvectors with different eigenvalues are orthogonal")
print("   (for symmetric matrices)")
print(f"   v₁ · v₂ = {np.dot(eigenvectors[:, 0], eigenvectors[:, 1]):.10f}")
print(f"   v₁ · v₃ = {np.dot(eigenvectors[:, 0], eigenvectors[:, 2]):.10f}")

# ============================================================================
# LESSON 8: Diagonalization - The Power of Eigenvectors
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 8: DIAGONALIZATION - Ultimate Simplification")
print("=" * 70)

print("\nIf A has n independent eigenvectors, we can write:")
print("A = PDP⁻¹")
print("where D is diagonal (eigenvalues) and P has eigenvectors as columns")

A = np.array([[3, 1],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

print("\nMatrix A:")
print(A)

print("\nP (eigenvectors as columns):")
print(P)

print("\nD (eigenvalues on diagonal):")
print(D)

print("\nP⁻¹:")
print(P_inv)

print("\nReconstruction: PDP⁻¹ =")
A_reconstructed = P @ D @ P_inv
print(A_reconstructed)

print("\nVerification (should be same as A):")
print(np.allclose(A, A_reconstructed))

# ============================================================================
# LESSON 9: Applications - Matrix Powers
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 9: APPLICATIONS - Computing A^n Efficiently")
print("=" * 70)

print("\nDiagonalization makes computing A^n easy:")
print("A^n = (PDP⁻¹)^n = PD^nP⁻¹")

A = np.array([[0.8, 0.3],
              [0.2, 0.7]])

print("\nMatrix A (transition matrix):")
print(A)

n = 10
print(f"\nComputing A^{n} using diagonalization:")

eigenvalues, eigenvectors = np.linalg.eig(A)
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

# Efficient: D^n
D_n = np.diag(eigenvalues**n)
A_n = P @ D_n @ P_inv

print(f"A^{n} =")
print(A_n)

print("\nVerification (direct computation):")
A_n_direct = np.linalg.matrix_power(A, n)
print(A_n_direct)
print(f"Methods agree: {np.allclose(A_n, A_n_direct)}")

# ============================================================================
# LESSON 10: Real-World Application - Markov Chains
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 10: MARKOV CHAINS - Finding Steady State")
print("=" * 70)

print("\nExample: Weather prediction")
print("Sunny tomorrow: 70% if sunny today, 40% if rainy today")
print("Rainy tomorrow: 30% if sunny today, 60% if rainy today")

P = np.array([[0.7, 0.4],  # Transition matrix
              [0.3, 0.6]])

print("\nTransition matrix P:")
print(P)
print("[Sunny, Rainy]")

# Find steady state (eigenvector for eigenvalue 1)
eigenvalues, eigenvectors = np.linalg.eig(P.T)  # Transpose for right eigenvector

# Find eigenvalue closest to 1
idx = np.argmin(np.abs(eigenvalues - 1))
steady_state = np.real(eigenvectors[:, idx])
steady_state = steady_state / np.sum(steady_state)  # Normalize

print(f"\nSteady state probabilities:")
print(f"Sunny: {steady_state[0]:.2%}")
print(f"Rainy: {steady_state[1]:.2%}")

print("\nVerification: After many steps, any initial state converges to steady state")
initial_state = np.array([1, 0])  # Start sunny
for day in [1, 5, 10, 20, 50]:
    state = np.linalg.matrix_power(P, day) @ initial_state
    print(f"Day {day:2d}: Sunny={state[0]:.4f}, Rainy={state[1]:.4f}")

# ============================================================================
# LESSON 11: PCA - Principal Component Analysis
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 11: PCA - Finding Important Directions in Data")
print("=" * 70)

print("\nPCA uses eigenvectors to find directions of maximum variance")

# Generate correlated data
np.random.seed(42)
X = np.random.randn(100, 2) @ np.array([[2, 1], [1, 1]])

print(f"\nData shape: {X.shape}")
print("Sample points:")
print(X[:5])

# Center the data
X_centered = X - np.mean(X, axis=0)

# Covariance matrix
cov_matrix = np.cov(X_centered.T)
print("\nCovariance matrix:")
print(cov_matrix)

# Eigenvectors are principal components
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nPrincipal components (eigenvectors):")
print(eigenvectors)
print("\nVariance explained (eigenvalues):")
print(eigenvalues)
print(f"Total variance: {np.sum(eigenvalues):.4f}")

print("\nPercentage of variance explained:")
for i, val in enumerate(eigenvalues):
    print(f"PC{i+1}: {val/np.sum(eigenvalues)*100:.2f}%")

# ============================================================================
# LESSON 12: Practice Problems
# ============================================================================
print("\n" + "=" * 70)
print("LESSON 12: PRACTICE EXERCISES")
print("=" * 70)

print("\nExercise 1: Find eigenvalues and eigenvectors")
A1 = np.array([[5, 2],
               [2, 5]])
print("\nMatrix:")
print(A1)
vals, vecs = np.linalg.eig(A1)
print(f"Eigenvalues: {vals}")
print("Eigenvectors:")
print(vecs)

print("\n" + "-" * 70)
print("Exercise 2: Verify A·v = λ·v")
v = vecs[:, 0]
lam = vals[0]
print(f"v = {v}")
print(f"λ = {lam}")
print(f"A·v = {A1 @ v}")
print(f"λ·v = {lam * v}")
print(f"Equal? {np.allclose(A1 @ v, lam * v)}")

print("\n" + "-" * 70)
print("Exercise 3: Diagonalize the matrix")
P = vecs
D = np.diag(vals)
print("P·D·P⁻¹ =")
print(P @ D @ np.linalg.inv(P))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("COURSE SUMMARY")
print("=" * 70)

summary = """
KEY CONCEPTS:

1. EIGENVECTOR: A vector that doesn't change direction under transformation
   A·v = λ·v

2. EIGENVALUE: The scaling factor (λ)

3. FINDING THEM: Solve det(A - λI) = 0

4. PROPERTIES:
   - Sum of eigenvalues = trace of matrix
   - Product of eigenvalues = determinant
   - Eigenvectors are orthogonal (for symmetric matrices)

5. DIAGONALIZATION: A = PDP⁻¹
   - Simplifies matrix operations
   - Makes computing A^n efficient

6. APPLICATIONS:
   - Markov chains (steady states)
   - PCA (data compression)
   - Differential equations
   - Google PageRank
   - Quantum mechanics
   - Vibration analysis

REMEMBER: Eigenvectors reveal the "natural directions" of a transformation!
"""

print(summary)

print("\n" + "=" * 70)
print("COURSE COMPLETE! 🎓")
print("=" * 70)