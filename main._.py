# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # For image loading and saving
import os  # For creating directories to save results

# === Part 1: Define Helper Functions === #

def initialize_centers_random(X, k):
    """
    Initialize cluster centers by randomly selecting k data points.

    Parameters:
    X (numpy.ndarray): The data points (pixels), shape (num_pixels, 3)
    k (int): Number of clusters

    Returns:
    centers (numpy.ndarray): Initialized cluster centers, shape (k, 3)
    """
    indices = np.random.choice(X.shape[0], k, replace=False)
    centers = X[indices]
    return centers

def initialize_centers_spaced(X, k):
    """
    Initialize cluster centers to be far apart from each other.

    Parameters:
    X (numpy.ndarray): The data points (pixels), shape (num_pixels, 3)
    k (int): Number of clusters

    Returns:
    centers (numpy.ndarray): Initialized cluster centers, shape (k, 3)
    """
    # Start by randomly selecting the first center
    centers = []
    centers.append(X[np.random.randint(0, X.shape[0])])

    # For each subsequent center
    for _ in range(1, k):
        # Compute distances from the existing centers
        dist_sq = np.array([min([np.inner(c - x, c - x) for c in centers]) for x in X])
        # Choose the next center with probability proportional to distance squared
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        index = np.searchsorted(cumulative_probs, r)
        centers.append(X[index])

    return np.array(centers)

def kmeans(X, k, init_strategy='random', max_iters=100):
    """
    Perform k-means clustering on data X.

    Parameters:
    X (numpy.ndarray): The data points (pixels), shape (num_pixels, 3)
    k (int): Number of clusters
    init_strategy (str): Initialization strategy ('random' or 'spaced')
    max_iters (int): Maximum number of iterations

    Returns:
    centers (numpy.ndarray): Final cluster centers, shape (k, 3)
    labels (numpy.ndarray): Cluster labels for each data point, shape (num_pixels,)
    num_iters (int): Number of iterations until convergence
    """
    # Initialize cluster centers
    if init_strategy == 'random':
        centers = initialize_centers_random(X, k)
    elif init_strategy == 'spaced':
        centers = initialize_centers_spaced(X, k)
    else:
        raise ValueError("Invalid initialization strategy")

    for i in range(max_iters):
        # Assign each data point to the closest center
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        # Compute new centers
        new_centers = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j] for j in range(k)])

        # Check for convergence
        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    return centers, labels, i + 1  # i + 1 because iteration starts from 0

def reconstruct_image(labels, centers, image_shape):
    """
    Reconstruct the image from labels and cluster centers.

    Parameters:
    labels (numpy.ndarray): Cluster labels for each data point, shape (num_pixels,)
    centers (numpy.ndarray): Cluster centers, shape (k, 3)
    image_shape (tuple): Original image shape (height, width, 3)

    Returns:
    reconstructed_image (numpy.ndarray): Reconstructed image array, shape (height, width, 3)
    """
    reconstructed_image = centers[labels].reshape(image_shape).astype(np.uint8)
    return reconstructed_image

def compute_mse(original_image, reconstructed_image):
    """
    Compute the Mean Squared Error between the original and reconstructed images.

    Parameters:
    original_image (numpy.ndarray): Original image array, shape (height, width, 3)
    reconstructed_image (numpy.ndarray): Reconstructed image array, shape (height, width, 3)

    Returns:
    mse (float): Mean Squared Error
    """
    mse = np.mean((original_image - reconstructed_image) ** 2)
    return mse

# === Part 2: Main Code to Run K-Means on Images === #

# List of k values to test
k_values = [2, 3, 10, 20, 40]

# Initialization strategies
# We will run the algorithm three times per k per image:
# Two times with 'random' initialization, and one time with 'spaced' initialization
init_strategies = ['random1', 'random2', 'spaced']  # Two random initializations and one spaced

# Load images of my choice
image_files = ['cat.jpg', 'car.jpg']
images = []
for image_file in image_files:
    image = np.array(Image.open(image_file))
    images.append(image)

# Create directories to save results
if not os.path.exists('results'):
    os.makedirs('results')

# For each image
for idx, image in enumerate(images):
    image_name = f'image_{idx+1}'
    print(f"\nProcessing {image_name}...")
    original_image = image
    image_shape = original_image.shape
    X = original_image.reshape(-1, 3)  # Reshape to (num_pixels, 3)

    # Create a result directory for the image
    result_dir = f'results/{image_name}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        # If directory exists, clear its contents to avoid appending to old results
        for file in os.listdir(result_dir):
            file_path = os.path.join(result_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    # Open the results file in write mode to overwrite existing content
    results_file = open(f'{result_dir}/results.txt', 'w')

    # For each k
    for k in k_values:
        # For each initialization strategy
        for init_idx, init_strategy in enumerate(init_strategies):
            print(f"  k = {k}, Initialization = {init_strategy}")

            # Ensure different random initializations
            if init_strategy.startswith('random'):
                np.random.seed()  # Reset seed to system time for randomness ( I did not include my seed so its random every time because the assignment instructions did not indicate to do so)
                init_type = 'random'
            else:
                init_type = init_strategy  # 'spaced'

            # Run k-means clustering
            centers, labels, num_iters = kmeans(X, k, init_strategy=init_type)

            # Reconstruct the image
            reconstructed_image = reconstruct_image(labels, centers, image_shape)

            # Compute MSE
            mse = compute_mse(original_image, reconstructed_image)
            print(f"    MSE: {mse:.2f}, Iterations: {num_iters}")

            # Save reconstructed image
            output_filename = f'{result_dir}/{image_name}_k{k}_init{init_idx+1}.png'
            Image.fromarray(reconstructed_image).save(output_filename)

            # Save MSE and number of iterations to the results file
            results_file.write(f'k={k}, init={init_strategy}, MSE={mse:.2f}, iterations={num_iters}\n')

    # Close the results file after processing all k values
    results_file.close()

