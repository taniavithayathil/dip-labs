"""
DIP MINI PROJECT
EXPERIMENT 4 - SPATIAL DOMAIN FILTERING

Tasks implemented:

TASK 1
- Select 3 project screenshots
- Add Salt-and-Pepper noise
- Add Gaussian noise
- Save Original -> Noisy comparisons
- Record observations

TASK 2
- Mean/Average Filter
- Gaussian Filter
- Median Filter
- 3x3 kernel
- 5x5 kernel
- Compare:
    * Noise reduction
    * Edge preservation
    * Blurring
    * Computational time
    * MSE
    * PSNR
    * Sharpness

Images used:
    ss_1.jpeg
    ss_2.jpeg
    ss_3.jpeg
"""

import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "original"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "task4"
)

NOISY_DIR = os.path.join(
    OUTPUT_DIR,
    "noisy"
)

MEAN_DIR = os.path.join(
    OUTPUT_DIR,
    "mean_filter"
)

GAUSSIAN_DIR = os.path.join(
    OUTPUT_DIR,
    "gaussian_filter"
)

MEDIAN_DIR = os.path.join(
    OUTPUT_DIR,
    "median_filter"
)

COMPARISON_DIR = os.path.join(
    OUTPUT_DIR,
    "comparisons"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

IMAGE_FILES = [
    "ss_1.jpeg",
    "ss_2.jpeg",
    "ss_3.jpeg"
]

KERNEL_SIZES = [3, 5]

# Salt-and-pepper noise amount
SALT_PEPPER_AMOUNT = 0.03

# Gaussian noise parameters
GAUSSIAN_MEAN = 0
GAUSSIAN_STD = 20

# Fixed seed so the same experiment can be reproduced
RANDOM_SEED = 42


# ============================================================
# DIRECTORY CREATION
# ============================================================

def create_directories():

    directories = [
        OUTPUT_DIR,
        NOISY_DIR,
        MEAN_DIR,
        GAUSSIAN_DIR,
        MEDIAN_DIR,
        COMPARISON_DIR
    ]

    for directory in directories:
        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Unable to load image:\n{path}"
        )

    # OpenCV loads BGR.
    # Convert to RGB for Matplotlib.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ============================================================
# IMAGE SAVING
# ============================================================

def save_image(path, image):

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

    # RGB -> BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        path,
        image_bgr
    )


# ============================================================
# TASK 1
# SALT-AND-PEPPER NOISE
# ============================================================

def add_salt_and_pepper_noise(
    image,
    amount=0.03,
    seed=42
):

    rng = np.random.default_rng(seed)

    noisy = image.copy()

    height, width = image.shape[:2]

    total_pixels = height * width

    noisy_pixels = int(
        total_pixels * amount
    )

    # Random pixel coordinates
    ys = rng.integers(
        0,
        height,
        noisy_pixels
    )

    xs = rng.integers(
        0,
        width,
        noisy_pixels
    )

    # Half salt, half pepper
    half = noisy_pixels // 2

    # Salt = white
    noisy[
        ys[:half],
        xs[:half]
    ] = 255

    # Pepper = black
    noisy[
        ys[half:],
        xs[half:]
    ] = 0

    return noisy


# ============================================================
# TASK 1
# GAUSSIAN NOISE
# ============================================================

def add_gaussian_noise(
    image,
    mean=0,
    std=20,
    seed=42
):

    rng = np.random.default_rng(seed)

    image_float = image.astype(
        np.float32
    )

    noise = rng.normal(
        mean,
        std,
        image.shape
    ).astype(
        np.float32
    )

    noisy = image_float + noise

    noisy = np.clip(
        noisy,
        0,
        255
    )

    return noisy.astype(
        np.uint8
    )


# ============================================================
# QUALITY METRICS
# ============================================================

def calculate_mse(
    original,
    processed
):

    original_float = original.astype(
        np.float64
    )

    processed_float = processed.astype(
        np.float64
    )

    mse = np.mean(
        (
            original_float -
            processed_float
        ) ** 2
    )

    return mse


def calculate_psnr(
    original,
    processed
):

    mse = calculate_mse(
        original,
        processed
    )

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10(
        (255.0 ** 2) / mse
    )

    return psnr


def calculate_sharpness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    return laplacian.var()


# ============================================================
# FILTER FUNCTIONS
# ============================================================

def mean_filter(
    image,
    kernel_size
):

    return cv2.blur(
        image,
        (
            kernel_size,
            kernel_size
        )
    )


def gaussian_filter(
    image,
    kernel_size
):

    return cv2.GaussianBlur(
        image,
        (
            kernel_size,
            kernel_size
        ),
        0
    )


def median_filter(
    image,
    kernel_size
):

    return cv2.medianBlur(
        image,
        kernel_size
    )


# ============================================================
# TIMING
# ============================================================

def apply_filter_with_timing(
    filter_function,
    image,
    kernel_size
):

    start = time.perf_counter()

    result = filter_function(
        image,
        kernel_size
    )

    end = time.perf_counter()

    execution_time = (
        end - start
    ) * 1000

    return result, execution_time


# ============================================================
# TASK 1 COMPARISON
# ============================================================

def create_noise_comparison(
    original,
    salt_pepper,
    gaussian,
    image_name,
    output_path
):

    plt.figure(
        figsize=(15, 5)
    )

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original
    )

    plt.title(
        "Original Image"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        salt_pepper
    )

    plt.title(
        "Salt-and-Pepper Noise"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        gaussian
    )

    plt.title(
        "Gaussian Noise"
    )

    plt.axis(
        "off"
    )

    plt.suptitle(
        f"Task 1 - Noise Comparison: {image_name}",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# TASK 2 COMPARISON
# ============================================================

def create_filter_comparison(
    original,
    noisy,
    mean_3,
    mean_5,
    gaussian_3,
    gaussian_5,
    median_3,
    median_5,
    image_name,
    noise_name,
    output_path
):

    images = [
        original,
        noisy,
        mean_3,
        mean_5,
        gaussian_3,
        gaussian_5,
        median_3,
        median_5
    ]

    titles = [
        "Original",
        f"Noisy - {noise_name}",
        "Mean 3x3",
        "Mean 5x5",
        "Gaussian 3x3",
        "Gaussian 5x5",
        "Median 3x3",
        "Median 5x5"
    ]

    plt.figure(
        figsize=(16, 9)
    )

    for i in range(8):

        plt.subplot(
            2,
            4,
            i + 1
        )

        plt.imshow(
            images[i]
        )

        plt.title(
            titles[i]
        )

        plt.axis(
            "off"
        )

    plt.suptitle(
        f"Spatial Domain Filtering - "
        f"{image_name} - {noise_name}",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# TASK 1 OBSERVATIONS
# ============================================================

def save_noise_observations(
    original,
    salt_pepper,
    gaussian,
    output_path
):

    mse_sp = calculate_mse(
        original,
        salt_pepper
    )

    psnr_sp = calculate_psnr(
        original,
        salt_pepper
    )

    sharpness_sp = calculate_sharpness(
        salt_pepper
    )

    mse_gaussian = calculate_mse(
        original,
        gaussian
    )

    psnr_gaussian = calculate_psnr(
        original,
        gaussian
    )

    sharpness_gaussian = calculate_sharpness(
        gaussian
    )

    original_sharpness = calculate_sharpness(
        original
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "TASK 1 - NOISY TEST IMAGE OBSERVATIONS\n"
        )

        file.write(
            "=======================================\n\n"
        )

        file.write(
            "SALT-AND-PEPPER NOISE\n"
        )

        file.write(
            f"MSE: {mse_sp:.4f}\n"
        )

        file.write(
            f"PSNR: {psnr_sp:.4f} dB\n"
        )

        file.write(
            f"Sharpness: {sharpness_sp:.4f}\n\n"
        )

        file.write(
            "GAUSSIAN NOISE\n"
        )

        file.write(
            f"MSE: {mse_gaussian:.4f}\n"
        )

        file.write(
            f"PSNR: {psnr_gaussian:.4f} dB\n"
        )

        file.write(
            f"Sharpness: {sharpness_gaussian:.4f}\n\n"
        )

        file.write(
            "VISUAL OBSERVATIONS\n"
        )

        file.write(
            "-------------------\n"
        )

        file.write(
            "Salt-and-pepper noise appears as "
            "isolated black and white pixels. "
            "It is impulse-type noise and is "
            "particularly visible around image "
            "details and uniform regions.\n\n"
        )

        file.write(
            "Gaussian noise appears as random "
            "intensity variations distributed "
            "throughout the image. It produces "
            "a grain-like appearance rather than "
            "isolated extreme pixels.\n\n"
        )

        file.write(
            f"Original sharpness: "
            f"{original_sharpness:.4f}\n"
        )

        file.write(
            "The numerical values can be used "
            "along with the visual comparison "
            "to evaluate the effect of noise."
        )


# ============================================================
# PROCESS ONE FILTER
# ============================================================

def process_filter(
    original,
    noisy,
    image_name,
    noise_name,
    filter_name,
    filter_function,
    output_directory
):

    results = {}

    for kernel_size in KERNEL_SIZES:

        result, execution_time = (
            apply_filter_with_timing(
                filter_function,
                noisy,
                kernel_size
            )
        )

        # Save filtered image
        filename = (
            f"{noise_name.lower().replace('-', '_')}_"
            f"{filter_name.lower()}_"
            f"{kernel_size}x{kernel_size}.png"
        )

        output_path = os.path.join(
            output_directory,
            filename
        )

        save_image(
            output_path,
            result
        )

        # Metrics
        mse = calculate_mse(
            original,
            result
        )

        psnr = calculate_psnr(
            original,
            result
        )

        sharpness = calculate_sharpness(
            result
        )

        results[kernel_size] = {
            "image": result,
            "mse": mse,
            "psnr": psnr,
            "sharpness": sharpness,
            "time": execution_time
        }

        print(
            f"{filter_name:10s} "
            f"{kernel_size}x{kernel_size} | "
            f"PSNR: {psnr:8.2f} dB | "
            f"Sharpness: {sharpness:10.2f} | "
            f"Time: {execution_time:8.3f} ms"
        )

    return results


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_file
):

    print("\n")
    print("=" * 70)

    print(
        f"PROCESSING {image_file}"
    )

    print("=" * 70)

    image_path = os.path.join(
        DATASET_DIR,
        image_file
    )

    original = load_image(
        image_path
    )

    image_name = os.path.splitext(
        image_file
    )[0]

    # --------------------------------------------------------
    # Create image-specific folders
    # --------------------------------------------------------

    noisy_image_dir = os.path.join(
        NOISY_DIR,
        image_name
    )

    mean_image_dir = os.path.join(
        MEAN_DIR,
        image_name
    )

    gaussian_image_dir = os.path.join(
        GAUSSIAN_DIR,
        image_name
    )

    median_image_dir = os.path.join(
        MEDIAN_DIR,
        image_name
    )

    os.makedirs(
        noisy_image_dir,
        exist_ok=True
    )

    os.makedirs(
        mean_image_dir,
        exist_ok=True
    )

    os.makedirs(
        gaussian_image_dir,
        exist_ok=True
    )

    os.makedirs(
        median_image_dir,
        exist_ok=True
    )

    # ========================================================
    # TASK 1 - NOISE
    # ========================================================

    print("\nTASK 1 - ADDING NOISE")

    salt_pepper = add_salt_and_pepper_noise(
        original,
        SALT_PEPPER_AMOUNT,
        RANDOM_SEED
    )

    gaussian_noise = add_gaussian_noise(
        original,
        GAUSSIAN_MEAN,
        GAUSSIAN_STD,
        RANDOM_SEED
    )

    # Save original
    save_image(
        os.path.join(
            noisy_image_dir,
            "original.png"
        ),
        original
    )

    # Save salt-and-pepper
    save_image(
        os.path.join(
            noisy_image_dir,
            "salt_pepper.png"
        ),
        salt_pepper
    )

    # Save Gaussian
    save_image(
        os.path.join(
            noisy_image_dir,
            "gaussian.png"
        ),
        gaussian_noise
    )

    # Save comparison
    create_noise_comparison(
        original,
        salt_pepper,
        gaussian_noise,
        image_name,
        os.path.join(
            noisy_image_dir,
            "comparison.png"
        )
    )

    # Save observations
    save_noise_observations(
        original,
        salt_pepper,
        gaussian_noise,
        os.path.join(
            noisy_image_dir,
            "observations.txt"
        )
    )

    print(
        "Noise generation completed."
    )

    # ========================================================
    # TASK 2 - FILTERING
    # ========================================================

    noise_images = {
        "Salt-and-Pepper": salt_pepper,
        "Gaussian": gaussian_noise
    }

    for noise_name, noisy_image in noise_images.items():

        print("\n")
        print(
            f"TASK 2 - {noise_name} NOISE"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # Mean filter
        # ----------------------------------------------------

        mean_results = process_filter(
            original,
            noisy_image,
            image_name,
            noise_name,
            "Mean",
            mean_filter,
            mean_image_dir
        )

        # ----------------------------------------------------
        # Gaussian filter
        # ----------------------------------------------------

        gaussian_results = process_filter(
            original,
            noisy_image,
            image_name,
            noise_name,
            "Gaussian",
            gaussian_filter,
            gaussian_image_dir
        )

        # ----------------------------------------------------
        # Median filter
        # ----------------------------------------------------

        median_results = process_filter(
            original,
            noisy_image,
            image_name,
            noise_name,
            "Median",
            median_filter,
            median_image_dir
        )

        # ----------------------------------------------------
        # Comparison figure
        # ----------------------------------------------------

        comparison_path = os.path.join(
            COMPARISON_DIR,
            f"{image_name}_"
            f"{noise_name.lower().replace('-', '_')}_"
            f"comparison.png"
        )

        create_filter_comparison(
            original,
            noisy_image,

            mean_results[3]["image"],
            mean_results[5]["image"],

            gaussian_results[3]["image"],
            gaussian_results[5]["image"],

            median_results[3]["image"],
            median_results[5]["image"],

            image_name,
            noise_name,
            comparison_path
        )

    print(
        f"\nCompleted {image_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("DIGITAL IMAGE PROCESSING")
    print("EXPERIMENT 4 - SPATIAL DOMAIN FILTERING")
    print("=" * 70)

    print(
        "\nImages selected:"
    )

    for image in IMAGE_FILES:
        print(
            f"  - {image}"
        )

    print(
        "\nFilters:"
    )

    print(
        "  - Mean/Average"
    )

    print(
        "  - Gaussian"
    )

    print(
        "  - Median"
    )

    print(
        "\nKernel sizes:"
    )

    print(
        "  - 3x3"
    )

    print(
        "  - 5x5"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Process all three screenshots
    # --------------------------------------------------------

    for image_file in IMAGE_FILES:

        process_image(
            image_file
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXPERIMENT 4 COMPLETED")
    print("=" * 70)

    print(
        f"\nAll outputs saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nGenerated:"
    )

    print(
        "  [1] Salt-and-pepper noisy images"
    )

    print(
        "  [2] Gaussian noisy images"
    )

    print(
        "  [3] Original vs noisy comparisons"
    )

    print(
        "  [4] Mean filter - 3x3 and 5x5"
    )

    print(
        "  [5] Gaussian filter - 3x3 and 5x5"
    )

    print(
        "  [6] Median filter - 3x3 and 5x5"
    )

    print(
        "  [7] Filter comparison figures"
    )

    print(
        "  [8] MSE / PSNR / sharpness measurements"
    )

    print(
        "  [9] Execution-time measurements"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()