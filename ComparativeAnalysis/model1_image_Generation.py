

import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


def apply_gaussian_edge_detection(image):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    return cv2.filter2D(image, -1, kernel)


def apply_sobel_edge_detection(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(grad_x, grad_y)


def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    angle = np.arctan2(grad_y, grad_x)
    nms = np.zeros_like(magnitude)
    angle = angle * 180.0 / np.pi
    angle[angle < 0] += 180
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            try:
                q = 255
                r = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]
                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    nms[i, j] = magnitude[i, j]
                else:
                    nms[i, j] = 0
            except IndexError as e:
                pass
    strong = 255
    weak = 50
    res = np.zeros_like(image)
    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms <= high_threshold) & (nms >= low_threshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    for i in range(1, res.shape[0] - 1):
        for j in range(1, res.shape[1] - 1):
            if res[i, j] == weak:
                if (
                    (res[i + 1, j - 1] == strong)
                    or (res[i + 1, j] == strong)
                    or (res[i + 1, j + 1] == strong)
                    or (res[i, j - 1] == strong)
                    or (res[i, j + 1] == strong)
                    or (res[i - 1, j - 1] == strong)
                    or (res[i - 1, j] == strong)
                    or (res[i - 1, j + 1] == strong)
                ):
                    res[i, j] = strong
                else:
                    res[i, j] = 0
    return res


def measure_performance(images, filenames):
    results = {"Filename": [], "Method": [], "Time": []}
    output_images = {"Gaussian": [], "Sobel": [], "Canny": []}
    for image, filename in zip(images, filenames):
        start_time = time.time()
        gaussian_edges = apply_gaussian_edge_detection(image)
        elapsed_time = time.time() - start_time
        results["Filename"].append(filename)
        results["Method"].append("Gaussian")
        results["Time"].append(elapsed_time)
        output_images["Gaussian"].append((filename, gaussian_edges))

        start_time = time.time()
        sobel_edges = apply_sobel_edge_detection(image)
        elapsed_time = time.time() - start_time
        results["Filename"].append(filename)
        results["Method"].append("Sobel")
        results["Time"].append(elapsed_time)
        output_images["Sobel"].append((filename, sobel_edges))

        start_time = time.time()
        canny_edges = apply_canny_edge_detection(image)
        elapsed_time = time.time() - start_time
        results["Filename"].append(filename)
        results["Method"].append("Canny")
        results["Time"].append(elapsed_time)
        output_images["Canny"].append((filename, canny_edges))
    return results, output_images


def save_results_to_csv(results, output_file):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


def save_output_images(output_images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for method, images in output_images.items():
        method_folder = os.path.join(output_folder, method)
        if not os.path.exists(method_folder):
            os.makedirs(method_folder)
        for filename, image in images:
            output_path = os.path.join(method_folder, filename)
            cv2.imwrite(output_path, image)


def plot_results(results):
    methods = ["Gaussian", "Sobel", "Canny"]
    times = [
        np.mean(
            [
                results["Time"][i]
                for i in range(len(results["Time"]))
                if results["Method"][i] == method
            ]
        )
        for method in methods
    ]
    plt.bar(methods, times)
    plt.xlabel("Edge Detection Method")
    plt.ylabel("Average Time (s)")
    plt.title("Performance Comparison of Edge Detection Methods")
    plt.show()


if __name__ == "__main__":
    folder = "input"
    images, filenames = load_images_from_folder(folder)
    results, output_images = measure_performance(images, filenames)
    save_results_to_csv(results, "performance_results.csv")
    save_output_images(output_images, "output")
    plot_results(results)
