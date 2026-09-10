"""
============================================================
DIGITAL IMAGE PROCESSING
SEMESTER MINI-PROJECT
EXPERIMENT 5 - SHARPENING AND FILTER INTEGRATION
============================================================

Project:
AI-Assisted Screenshot Authenticity Detection Using
Digital Image Processing

Experiment 5 Tasks:

Task 1 - Implement Sharpening Filters
    1. Laplacian filter
    2. Gradient-based sharpening
    3. High-boost filtering

Task 2 - Compare Different Filters
    Mean
    Gaussian
    Median
    Laplacian
    High-Boost

Task 3 - Select the Best Filter

Task 4 - Integrate Filtering Module
    Interactive filter selection

Task 5 - Compare Enhancement + Filtering

Task 6 - Generate documentation-supporting outputs

Previous experiments are retained:
    Experiment 3 -> src/task3/enhancement.py
    Experiment 4 -> src/task4/filtering.py

This module extends the existing project and does NOT
replace previous modules.
"""

import os
import csv
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
    "task5"
)

SHARPENING_DIR = os.path.join(
    OUTPUT_DIR,
    "sharpening"
)

COMPARISON_DIR = os.path.join(
    OUTPUT_DIR,
    "filter_comparison"
)

ENHANCEMENT_FILTERING_DIR = os.path.join(
    OUTPUT_DIR,
    "enhancement_filtering"
)

TABLE_DIR = os.path.join(
    OUTPUT_DIR,
    "tables"
)


# ============================================================
# DATASET
# ============================================================

IMAGE_FILES = [
    "ss_1.jpeg",
    "ss_2.jpeg",
    "ss_3.jpeg"
]


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

KERNEL_SIZES = [3, 5]

# High-boost amplification factor
HIGH_BOOST_A = 1.5

# Laplacian strength
LAPLACIAN_ALPHA = 1.0

# Gradient sharpening strength
GRADIENT_ALPHA = 1.0

# Noise parameters retained from Experiment 4
SALT_PEPPER_AMOUNT = 0.03
GAUSSIAN_MEAN = 0
GAUSSIAN_STD = 20

RANDOM_SEED = 42


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    directories = [
        OUTPUT_DIR,
        SHARPENING_DIR,
        COMPARISON_DIR,
        ENHANCEMENT_FILTERING_DIR,
        TABLE_DIR
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
            f"Could not load image:\n{path}"
        )

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
    ).astype(
        np.uint8
    )

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        path,
        image_bgr
    )


# ============================================================
# NOISE FUNCTIONS
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

    number_of_pixels = int(
        total_pixels * amount
    )

    ys = rng.integers(
        0,
        height,
        number_of_pixels
    )

    xs = rng.integers(
        0,
        width,
        number_of_pixels
    )

    half = number_of_pixels // 2

    # Salt
    noisy[
        ys[:half],
        xs[:half]
    ] = 255

    # Pepper
    noisy[
        ys[half:],
        xs[half:]
    ] = 0

    return noisy


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
# BASIC SMOOTHING FILTERS
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
# TASK 1 - LAPLACIAN SHARPENING
# ============================================================

def laplacian_sharpen(
    image,
    alpha=1.0
):

    image_float = image.astype(
        np.float32
    )

    # OpenCV Laplacian
    laplacian = cv2.Laplacian(
        image_float,
        cv2.CV_32F,
        ksize=3
    )

    # Sharpening:
    # g(x,y) = f(x,y) - alpha * Laplacian(f)
    sharpened = (
        image_float -
        alpha * laplacian
    )

    sharpened = np.clip(
        sharpened,
        0,
        255
    )

    return sharpened.astype(
        np.uint8
    )


# ============================================================
# TASK 1 - GRADIENT-BASED SHARPENING
# ============================================================

def gradient_sharpen(
    image,
    alpha=1.0
):

    image_float = image.astype(
        np.float32
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    gray_float = gray.astype(
        np.float32
    )

    # Sobel gradients
    gx = cv2.Sobel(
        gray_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Gradient magnitude
    magnitude = cv2.magnitude(
        gx,
        gy
    )

    # Normalize gradient magnitude
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Convert gradient magnitude into
    # 3-channel image
    gradient_3channel = cv2.cvtColor(
        magnitude.astype(np.uint8),
        cv2.COLOR_GRAY2RGB
    ).astype(
        np.float32
    )

    # Add edge information to original
    sharpened = (
        image_float +
        alpha * gradient_3channel
    )

    sharpened = np.clip(
        sharpened,
        0,
        255
    )

    return sharpened.astype(
        np.uint8
    )


# ============================================================
# TASK 1 - HIGH-BOOST FILTERING
# ============================================================

def high_boost_filter(
    image,
    A=1.5,
    kernel_size=3
):

    image_float = image.astype(
        np.float32
    )

    # Low-pass version
    blurred = cv2.GaussianBlur(
        image_float,
        (
            kernel_size,
            kernel_size
        ),
        0
    )

    # High-frequency mask
    high_frequency = (
        image_float -
        blurred
    )

    # High-boost:
    # g = A*f - lowpass(f)
    #
    # Equivalent:
    # g = f + (A-1) * high_frequency

    sharpened = (
        A * image_float -
        blurred
    )

    sharpened = np.clip(
        sharpened,
        0,
        255
    )

    return sharpened.astype(
        np.uint8
    )


# ============================================================
# METRICS
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

    return np.mean(
        (
            original_float -
            processed_float
        ) ** 2
    )


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

    return (
        10 *
        np.log10(
            (255.0 ** 2) /
            mse
        )
    )


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
# TIMING
# ============================================================

def timed_operation(
    function,
    image,
    *args
):

    start = time.perf_counter()

    result = function(
        image,
        *args
    )

    end = time.perf_counter()

    elapsed_ms = (
        end - start
    ) * 1000

    return result, elapsed_ms


# ============================================================
# VISUAL COMPARISON - SHARPENING
# ============================================================

def create_sharpening_comparison(
    original,
    laplacian,
    gradient,
    high_boost,
    image_name,
    output_path
):

    images = [
        original,
        laplacian,
        gradient,
        high_boost
    ]

    titles = [
        "Original",
        "Laplacian Sharpening",
        "Gradient-Based Sharpening",
        f"High-Boost (A={HIGH_BOOST_A})"
    ]

    plt.figure(
        figsize=(16, 5)
    )

    for i in range(4):

        plt.subplot(
            1,
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
        f"Experiment 5 - Sharpening Comparison: "
        f"{image_name}",
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
# COMPLETE FILTER COMPARISON
# ============================================================

def create_filter_comparison(
    original,
    mean_result,
    gaussian_result,
    median_result,
    laplacian_result,
    high_boost_result,
    image_name,
    output_path
):

    images = [
        original,
        mean_result,
        gaussian_result,
        median_result,
        laplacian_result,
        high_boost_result
    ]

    titles = [
        "Original",
        "Mean Filter",
        "Gaussian Filter",
        "Median Filter",
        "Laplacian",
        "High-Boost"
    ]

    plt.figure(
        figsize=(18, 7)
    )

    for i in range(6):

        plt.subplot(
            2,
            3,
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
        f"Experiment 5 - Filter Comparison: "
        f"{image_name}",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# TASK 5 - ENHANCEMENT
# ============================================================

def gamma_enhancement(
    image,
    gamma=0.5
):

    image_float = image.astype(
        np.float32
    ) / 255.0

    enhanced = np.power(
        image_float,
        gamma
    )

    enhanced = (
        enhanced * 255
    )

    return np.clip(
        enhanced,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# TASK 5 - ENHANCEMENT + FILTERING
# ============================================================

def create_enhancement_filtering_comparison(
    original,
    enhanced,
    filtered,
    image_name,
    output_path
):

    images = [
        original,
        enhanced,
        filtered
    ]

    titles = [
        "Original Screenshot",
        "Enhanced Image\n(Gamma = 0.5)",
        "Enhanced + Median Filter"
    ]

    plt.figure(
        figsize=(15, 5)
    )

    for i in range(3):

        plt.subplot(
            1,
            3,
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
        f"Enhancement + Filtering Pipeline: "
        f"{image_name}",
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
# CSV SETUP
# ============================================================

def initialize_csv(
    csv_path
):

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "Image",
            "Filter",
            "Kernel / Parameter",
            "MSE",
            "PSNR (dB)",
            "Sharpness",
            "Execution Time (ms)"
        ])


def append_result(
    csv_path,
    image_name,
    filter_name,
    parameter,
    mse,
    psnr,
    sharpness,
    execution_time
):

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            image_name,
            filter_name,
            parameter,
            f"{mse:.6f}",
            f"{psnr:.6f}",
            f"{sharpness:.6f}",
            f"{execution_time:.6f}"
        ])


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_file,
    csv_path
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

    image_sharpening_dir = os.path.join(
        SHARPENING_DIR,
        image_name
    )

    image_comparison_dir = os.path.join(
        COMPARISON_DIR,
        image_name
    )

    image_pipeline_dir = os.path.join(
        ENHANCEMENT_FILTERING_DIR,
        image_name
    )

    os.makedirs(
        image_sharpening_dir,
        exist_ok=True
    )

    os.makedirs(
        image_comparison_dir,
        exist_ok=True
    )

    os.makedirs(
        image_pipeline_dir,
        exist_ok=True
    )

    # ========================================================
    # TASK 1 - SHARPENING
    # ========================================================

    print("\nTASK 1 - SHARPENING FILTERS")

    # --------------------------------------------------------
    # Laplacian
    # --------------------------------------------------------

    laplacian, lap_time = timed_operation(
        laplacian_sharpen,
        original,
        LAPLACIAN_ALPHA
    )

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    gradient, gradient_time = timed_operation(
        gradient_sharpen,
        original,
        GRADIENT_ALPHA
    )

    # --------------------------------------------------------
    # High Boost
    # --------------------------------------------------------

    high_boost, high_boost_time = timed_operation(
        high_boost_filter,
        original,
        HIGH_BOOST_A,
        3
    )

    # Save sharpening outputs
    save_image(
        os.path.join(
            image_sharpening_dir,
            "original.png"
        ),
        original
    )

    save_image(
        os.path.join(
            image_sharpening_dir,
            "laplacian.png"
        ),
        laplacian
    )

    save_image(
        os.path.join(
            image_sharpening_dir,
            "gradient_sharpening.png"
        ),
        gradient
    )

    save_image(
        os.path.join(
            image_sharpening_dir,
            "high_boost.png"
        ),
        high_boost
    )

    # Comparison
    create_sharpening_comparison(
        original,
        laplacian,
        gradient,
        high_boost,
        image_name,
        os.path.join(
            image_comparison_dir,
            "sharpening_comparison.png"
        )
    )

    # Metrics for sharpening
    sharpening_results = [
        (
            "Laplacian",
            "3x3",
            laplacian,
            lap_time
        ),
        (
            "Gradient",
            "Sobel 3x3",
            gradient,
            gradient_time
        ),
        (
            "High-Boost",
            f"A={HIGH_BOOST_A}",
            high_boost,
            high_boost_time
        )
    ]

    for (
        filter_name,
        parameter,
        result,
        execution_time
    ) in sharpening_results:

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

        append_result(
            csv_path,
            image_name,
            filter_name,
            parameter,
            mse,
            psnr,
            sharpness,
            execution_time
        )

        print(
            f"{filter_name:12s} | "
            f"PSNR: {psnr:8.2f} dB | "
            f"Sharpness: {sharpness:10.2f} | "
            f"Time: {execution_time:.3f} ms"
        )

    # ========================================================
    # TASK 2 - FILTER COMPARISON
    # ========================================================

    print("\nTASK 2 - FILTER COMPARISON")

    # Use 3x3 as the common comparison kernel
    mean_result, mean_time = timed_operation(
        mean_filter,
        original,
        3
    )

    gaussian_result, gaussian_time = timed_operation(
        gaussian_filter,
        original,
        3
    )

    median_result, median_time = timed_operation(
        median_filter,
        original,
        3
    )

    create_filter_comparison(
        original,
        mean_result,
        gaussian_result,
        median_result,
        laplacian,
        high_boost,
        image_name,
        os.path.join(
            image_comparison_dir,
            "complete_filter_comparison.png"
        )
    )

    comparison_results = [
        (
            "Mean",
            "3x3",
            mean_result,
            mean_time
        ),
        (
            "Gaussian",
            "3x3",
            gaussian_result,
            gaussian_time
        ),
        (
            "Median",
            "3x3",
            median_result,
            median_time
        ),
        (
            "Laplacian",
            "3x3",
            laplacian,
            lap_time
        ),
        (
            "High-Boost",
            f"A={HIGH_BOOST_A}",
            high_boost,
            high_boost_time
        )
    ]

    for (
        filter_name,
        parameter,
        result,
        execution_time
    ) in comparison_results:

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

        append_result(
            csv_path,
            image_name,
            f"Comparison-{filter_name}",
            parameter,
            mse,
            psnr,
            sharpness,
            execution_time
        )

    # ========================================================
    # TEST DIFFERENT KERNEL SIZES
    # ========================================================

    print("\nKERNEL SIZE COMPARISON")

    for kernel_size in KERNEL_SIZES:

        mean_result, mean_time = timed_operation(
            mean_filter,
            original,
            kernel_size
        )

        gaussian_result, gaussian_time = timed_operation(
            gaussian_filter,
            original,
            kernel_size
        )

        median_result, median_time = timed_operation(
            median_filter,
            original,
            kernel_size
        )

        print(
            f"\nKernel: {kernel_size}x{kernel_size}"
        )

        for (
            name,
            result,
            execution_time
        ) in [
            (
                "Mean",
                mean_result,
                mean_time
            ),
            (
                "Gaussian",
                gaussian_result,
                gaussian_time
            ),
            (
                "Median",
                median_result,
                median_time
            )
        ]:

            psnr = calculate_psnr(
                original,
                result
            )

            print(
                f"{name:10s} -> "
                f"PSNR: {psnr:.2f} dB | "
                f"Time: {execution_time:.3f} ms"
            )

    # ========================================================
    # TASK 5 - ENHANCEMENT + FILTERING
    # ========================================================

    print("\nTASK 5 - ENHANCEMENT + FILTERING")

    # Enhancement stage
    enhanced = gamma_enhancement(
        original,
        gamma=0.5
    )

    # Filtering stage
    final_filtered = median_filter(
        enhanced,
        3
    )

    # Save intermediate and final results
    save_image(
        os.path.join(
            image_pipeline_dir,
            "original.png"
        ),
        original
    )

    save_image(
        os.path.join(
            image_pipeline_dir,
            "enhanced_gamma_0.5.png"
        ),
        enhanced
    )

    save_image(
        os.path.join(
            image_pipeline_dir,
            "final_enhanced_median.png"
        ),
        final_filtered
    )

    # Create final comparison
    create_enhancement_filtering_comparison(
        original,
        enhanced,
        final_filtered,
        image_name,
        os.path.join(
            image_pipeline_dir,
            "pipeline_comparison.png"
        )
    )

    # Metrics
    enhanced_psnr = calculate_psnr(
        original,
        enhanced
    )

    final_psnr = calculate_psnr(
        original,
        final_filtered
    )

    print(
        f"Enhanced image PSNR: "
        f"{enhanced_psnr:.2f} dB"
    )

    print(
        f"Final enhanced + median PSNR: "
        f"{final_psnr:.2f} dB"
    )

    print(
        "\nCompleted:",
        image_file
    )


# ============================================================
# TASK 4 - INTERACTIVE FILTERING MODULE
# ============================================================

def interactive_filtering():

    print("\n")
    print("=" * 70)
    print("INTERACTIVE SPATIAL FILTERING MODULE")
    print("=" * 70)

    print(
        "\nAvailable images:"
    )

    for index, filename in enumerate(
        IMAGE_FILES,
        start=1
    ):

        print(
            f"{index}. {filename}"
        )

    try:

        image_choice = int(
            input(
                "\nSelect image (1-3): "
            )
        )

        if image_choice not in [1, 2, 3]:

            print(
                "Invalid image selection."
            )

            return

    except ValueError:

        print(
            "Invalid input."
        )

        return

    selected_file = IMAGE_FILES[
        image_choice - 1
    ]

    image_path = os.path.join(
        DATASET_DIR,
        selected_file
    )

    original = load_image(
        image_path
    )

    print("\n")
    print(
        "FILTER MENU"
    )

    print(
        "------------"
    )

    print(
        "1. Mean Filter"
    )

    print(
        "2. Gaussian Filter"
    )

    print(
        "3. Median Filter"
    )

    print(
        "4. Laplacian Sharpening"
    )

    print(
        "5. Gradient-Based Sharpening"
    )

    print(
        "6. High-Boost Filtering"
    )

    print(
        "7. Show All Filters"
    )

    try:

        choice = int(
            input(
                "\nSelect filter (1-7): "
            )
        )

    except ValueError:

        print(
            "Invalid input."
        )

        return

    result = None

    filter_name = ""

    if choice == 1:

        result = mean_filter(
            original,
            3
        )

        filter_name = "Mean 3x3"

    elif choice == 2:

        result = gaussian_filter(
            original,
            3
        )

        filter_name = "Gaussian 3x3"

    elif choice == 3:

        result = median_filter(
            original,
            3
        )

        filter_name = "Median 3x3"

    elif choice == 4:

        result = laplacian_sharpen(
            original
        )

        filter_name = "Laplacian"

    elif choice == 5:

        result = gradient_sharpen(
            original
        )

        filter_name = "Gradient-Based"

    elif choice == 6:

        result = high_boost_filter(
            original,
            HIGH_BOOST_A,
            3
        )

        filter_name = (
            f"High-Boost A={HIGH_BOOST_A}"
        )

    elif choice == 7:

        mean_result = mean_filter(
            original,
            3
        )

        gaussian_result = gaussian_filter(
            original,
            3
        )

        median_result = median_filter(
            original,
            3
        )

        laplacian_result = laplacian_sharpen(
            original
        )

        gradient_result = gradient_sharpen(
            original
        )

        high_boost_result = high_boost_filter(
            original,
            HIGH_BOOST_A,
            3
        )

        images = [
            original,
            mean_result,
            gaussian_result,
            median_result,
            laplacian_result,
            gradient_result,
            high_boost_result
        ]

        titles = [
            "Original",
            "Mean",
            "Gaussian",
            "Median",
            "Laplacian",
            "Gradient",
            "High-Boost"
        ]

        plt.figure(
            figsize=(16, 10)
        )

        for i in range(7):

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
            "Interactive Filter Comparison"
        )

        plt.tight_layout()

        plt.show()

        return

    else:

        print(
            "Invalid filter selection."
        )

        return

    # Show selected result
    plt.figure(
        figsize=(10, 5)
    )

    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        original
    )

    plt.title(
        "Original"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        result
    )

    plt.title(
        filter_name
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# GENERATE ALL EXPERIMENT OUTPUTS
# ============================================================

def run_experiment():

    create_directories()

    csv_path = os.path.join(
        OUTPUT_DIR,
        "experiment5_results.csv"
    )

    initialize_csv(
        csv_path
    )

    print("\n")
    print("=" * 70)
    print("DIGITAL IMAGE PROCESSING")
    print("EXPERIMENT 5")
    print("SHARPENING AND FILTER INTEGRATION")
    print("=" * 70)

    print(
        "\nProject pipeline:"
    )

    print(
        "Image Acquisition"
    )

    print(
        "       ↓"
    )

    print(
        "Image Enhancement"
    )

    print(
        "       ↓"
    )

    print(
        "Spatial Domain Filtering"
    )

    print(
        "       ↓"
    )

    print(
        "Segmentation"
    )

    print(
        "       ↓"
    )

    print(
        "Authenticity Analysis"
    )

    for image_file in IMAGE_FILES:

        process_image(
            image_file,
            csv_path
        )

    print("\n")
    print("=" * 70)
    print("EXPERIMENT 5 OUTPUT GENERATION COMPLETED")
    print("=" * 70)

    print(
        f"\nResults saved to:\n{OUTPUT_DIR}"
    )

    print(
        "\nCSV file:"
    )

    print(
        csv_path
    )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("DIP MINI-PROJECT - EXPERIMENT 5")
    print("=" * 70)

    print(
        "\n1. Generate all Experiment 5 outputs"
    )

    print(
        "2. Open interactive filtering module"
    )

    print(
        "3. Generate outputs and open interactive module"
    )

    print(
        "4. Exit"
    )

    try:

        choice = int(
            input(
                "\nEnter your choice: "
            )
        )

    except ValueError:

        print(
            "Invalid input."
        )

        return

    if choice == 1:

        run_experiment()

    elif choice == 2:

        interactive_filtering()

    elif choice == 3:

        run_experiment()

        interactive_filtering()

    elif choice == 4:

        print(
            "Exiting."
        )

    else:

        print(
            "Invalid choice."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()