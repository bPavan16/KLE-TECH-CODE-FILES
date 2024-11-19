


import cv2
import numpy as np
import os
import time
from skimage.metrics import structural_similarity as ssim

# Paths
images_path = "input"
output_path = "output_edge_detection"


# Function to load images
def load_images(path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((filename, img))
    return images


# Edge Detection Methods
def apply_gaussian_edge_detection(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges


def apply_sobel_edge_detection(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(grad_x, grad_y)
    _, edges = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)
    return edges


def apply_canny_edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges


# Noise Addition Function
def add_noise(image, mean=0, var=0.01):
    row, col = image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_image = np.clip(image + gauss * 255, 0, 255).astype(np.uint8)
    return noisy_image


# Performance Metrics
def compute_metrics(original, edge_image):
    accuracy = np.sum(original == edge_image) / original.size
    return accuracy


# Main Analysis
def analyze_edge_detection(images):
    results = []
    for filename, image in images:
        # Add noise to test noise robustness
        noisy_image = add_noise(image)

        # Apply edge detection methods
        methods = {
            "Gaussian": apply_gaussian_edge_detection,
            "Sobel": apply_sobel_edge_detection,
            "Canny": apply_canny_edge_detection,
        }
        for method_name, method in methods.items():
            start_time = time.time()
            edges = method(image)
            noisy_edges = method(noisy_image)
            time_taken = time.time() - start_time

            # Accuracy (SSIM for qualitative edge comparison if no ground truth)
            accuracy = ssim(image, edges, data_range=image.max() - image.min())
            noise_robustness = ssim(
                edges, noisy_edges, data_range=edges.max() - edges.min()
            )

            results.append(
                {
                    "Image": filename,
                    "Method": method_name,
                    "Time": time_taken,
                    "Accuracy": accuracy,
                    "Noise Robustness": noise_robustness,
                }
            )

    return results


# Load images and analyze
images = load_images(images_path)
results = analyze_edge_detection(images)

# Report Results
import pandas as pd

df = pd.DataFrame(results)
print(df)

# Save results to a CSV file
df.to_csv(
    os.path.join(output_path, "edge_detection_comparative_analysis.csv"), index=False
)
