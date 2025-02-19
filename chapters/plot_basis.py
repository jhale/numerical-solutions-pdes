import numpy as np
import matplotlib.pyplot as plt

def generate_mesh(N):
    """Generate a uniform mesh on [0, 1] with N nodes."""
    return np.linspace(0.0, 1.0, N)

def p1_basis_function(x, nodes, j):
    """Evaluate the j-th P1 basis function at points x."""
    phi = np.zeros_like(x)
    if j > 0:
        left = (x >= nodes[j - 1]) & (x <= nodes[j])
        phi[left] = (x[left] - nodes[j - 1]) / (nodes[j] - nodes[j - 1])
    if j < len(nodes) - 1:
        right = (x >= nodes[j]) & (x <= nodes[j + 1])
        phi[right] = (nodes[j + 1] - x[right]) / (nodes[j + 1] - nodes[j])
    return phi

def plot_p1_basis(N):
    """Plot all P1 basis functions on a mesh with N nodes."""
    resolution = 500
    nodes = generate_mesh(N)
    x = np.linspace(0, 1, resolution)
    
    plt.figure()
    for j in range(N):
        phi = p1_basis_function(x, nodes, j)
        line, = plt.plot(x, phi, label=f"$\\varphi_{j}(x)$")
        plt.scatter(nodes[j], 0.0, color=line.get_color())

    # Plot vertical lines at element boundaries and label elements
    for i in range(N - 1):
        plt.axvline(nodes[i], color='gray', linestyle='-', alpha=0.6)
        plt.axvline(nodes[i + 1], color='gray', linestyle='-', alpha=0.6)
        # Place label in the middle of the element
        mid_point = (nodes[i] + nodes[i + 1]) / 2
        plt.text(mid_point, -0.1, f"$K_{{{i}}}$", ha='center', va='center', fontsize=10)

    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$\\varphi_j(x)$", fontsize=12)
    plt.xticks(nodes)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right", fontsize=9)
    plt.show()

if __name__ == "__main__":
    N = 6  # Number of nodes (adjust as desired)
    plot_p1_basis(N)