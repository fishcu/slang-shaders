import numpy as np

# Original 8x8 Bayer matrix
bayer_matrix_8x8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]
])

# Normalize the matrix to the range 
# bayer_matrix_8x8 = bayer_matrix_8x8 / 64.0

# Function to split the 8x8 matrix into 4 submatrices based on the given order
def split_matrix(matrix, order):
    submatrices = []
    for i in range(len(order)):
        if order[i] == 0:  # top-left
            submatrix = matrix[0:4, 0:8]
        elif order[i] == 1:  # bottom-left
            submatrix = matrix[4:8, 0:8]
        elif order[i] == 2:  # top-right
            submatrix = matrix[0:4, 4:8]
        else:  # bottom-right
            submatrix = matrix[4:8, 4:8]
        submatrices.append(submatrix)
    return submatrices

# Function to flatten the 4x4x4 matrix into a 1D array
def flatten_matrix(submatrices):
    flat_matrix = np.array([])
    for submatrix in submatrices:
        flat_matrix = np.concatenate((flat_matrix, submatrix.flatten()))
    return flat_matrix

# Function to format the flattened matrix for GLSL
def format_matrix_for_glsl(flat_matrix):
    formatted_matrix = "const float bayer_4x4x4[64] = float[](\n    "
    for i, value in enumerate(flat_matrix):
        formatted_matrix += f"{value:.1f}/64.0"
        if i < len(flat_matrix) - 1:
            formatted_matrix += ", "
        if (i + 1) % 8 == 0:
            formatted_matrix += "\n    "
    formatted_matrix += "\n);"
    return formatted_matrix

# Example usage
# Define the order of the submatrices (0: top-left, 1: bottom-left, 2: top-right, 3: bottom-right)
# order = [0, 1, 2, 3]  # You can change this to any order you want
order = [0, 1]

# Split the 8x8 matrix into 4 submatrices based on the given order
submatrices = split_matrix(bayer_matrix_8x8, order)

# Flatten the 4x4x4 matrix into a 1D array
flat_matrix = flatten_matrix(submatrices)

# Format the flattened matrix for GLSL
formatted_matrix = format_matrix_for_glsl(flat_matrix)

# Print the formatted matrix
print(formatted_matrix)
