"""
Experiment 6 - Image Segmentation: Edge Detection and Thresholding

Project:
AI-Assisted Screenshot Authenticity Detection Using Digital Image Processing

Pipeline:
Image Acquisition -> Enhancement -> Filtering -> Segmentation

Tasks:
1. Prepare input image using the best output from Experiment 5
2. Edge Detection:
   - Roberts
   - Prewitt
   - Sobel
   - Laplacian
3. Thresholding:
   - Global Thresholding
   - Otsu Thresholding
   - Adaptive Thresholding
4. Analyse segmentation results
5. Select the most suitable segmentation technique
6. Save results and comparison data
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "original")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "task6")

INPUT_DIR = os.path.join(OUTPUT_DIR, "input")

EDGE_DIR = os.path.join(OUTPUT_DIR, "edges")
ROBERTS_DIR = os.path.join(EDGE_DIR, "roberts")
PREWITT_DIR = os.path.join(EDGE_DIR, "prewitt")
SOBEL_DIR = os.path.join(EDGE_DIR, "sobel")
LAPLACIAN_DIR = os.path.join(EDGE_DIR, "laplacian")

THRESHOLD_DIR = os.path.join(OUTPUT_DIR, "threshold")
GLOBAL_DIR = os.path.join(THRESHOLD_DIR, "global")
OTSU_DIR = os.path.join(THRESHOLD_DIR, "otsu")
ADAPTIVE_DIR = os.path.join(THRESHOLD_DIR, "adaptive")

COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparisons")

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "experiment6_results.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DIRECTORIES = [
    OUTPUT_DIR,
    INPUT_DIR,
    EDGE_DIR,
    ROBERTS_DIR,
    PREWITT_DIR,
    SOBEL_DIR,
    LAPLACIAN_DIR,
    THRESHOLD_DIR,
    GLOBAL_DIR,
    OTSU_DIR,
    ADAPTIVE_DIR,
    COMPARISON_DIR,
]

for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

IMAGE_NAMES = [
    "ss_1.jpeg",
    "ss_2.jpeg",
    "ss_3.jpeg",
]


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_name):
    """
    Load an image from the project dataset.
    """

    path = os.path.join(DATASET_DIR, image_name)

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    return image


# ============================================================
# EXPERIMENT 5 BEST OUTPUT
# ============================================================

def prepare_best_input(image):
    """
    Reproduce the best preprocessing pipeline selected
    in Experiment 5:

        Original
            ↓
        Gamma Enhancement (gamma = 0.5)
            ↓
        Median Filter (3x3)

    This produces the improved image used for segmentation.
    """

    # Convert to float for gamma transformation
    normalized = image.astype(np.float32) / 255.0

    gamma = 0.5

    gamma_corrected = np.power(
        normalized,
        gamma
    )

    gamma_corrected = np.uint8(
        gamma_corrected * 255
    )

    # Median filtering
    filtered = cv2.medianBlur(
        gamma_corrected,
        3
    )

    return filtered


# ============================================================
# ROBERTS OPERATOR
# ============================================================

def roberts_operator(gray):
    """
    Roberts cross-gradient operator.

    Uses two 2x2 kernels.
    """

    kernel_x = np.array(
        [
            [1, 0],
            [0, -1]
        ],
        dtype=np.float32
    )

    kernel_y = np.array(
        [
            [0, 1],
            [-1, 0]
        ],
        dtype=np.float32
    )

    gx = cv2.filter2D(
        gray,
        cv2.CV_32F,
        kernel_x
    )

    gy = cv2.filter2D(
        gray,
        cv2.CV_32F,
        kernel_y
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(magnitude)


# ============================================================
# PREWITT OPERATOR
# ============================================================

def prewitt_operator(gray):
    """
    Prewitt edge detector using 3x3 kernels.
    """

    kernel_x = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        dtype=np.float32
    )

    kernel_y = np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ],
        dtype=np.float32
    )

    gx = cv2.filter2D(
        gray,
        cv2.CV_32F,
        kernel_x
    )

    gy = cv2.filter2D(
        gray,
        cv2.CV_32F,
        kernel_y
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(magnitude)


# ============================================================
# SOBEL OPERATOR
# ============================================================

def sobel_operator(gray):
    """
    Sobel edge detector using 3x3 kernels.
    """

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(magnitude)


# ============================================================
# LAPLACIAN OPERATOR
# ============================================================

def laplacian_operator(gray):
    """
    Laplacian second-order derivative operator.
    """

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_32F,
        ksize=3
    )

    absolute_laplacian = np.abs(
        laplacian
    )

    absolute_laplacian = cv2.normalize(
        absolute_laplacian,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(
        absolute_laplacian
    )


# ============================================================
# THRESHOLDING METHODS
# ============================================================

def global_threshold(gray):
    """
    Manual global thresholding.

    Threshold value selected for the screenshot dataset.
    """

    threshold_value = 128

    _, binary = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    return binary, threshold_value


def otsu_threshold(gray):
    """
    Automatic Otsu thresholding.
    """

    threshold_value, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary, threshold_value


def adaptive_threshold(gray):
    """
    Adaptive Gaussian thresholding.

    Useful when illumination/background intensity
    is not uniform.
    """

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return binary


# ============================================================
# IMAGE QUALITY / SEGMENTATION METRICS
# ============================================================

def foreground_percentage(binary):
    """
    Percentage of pixels classified as foreground.
    """

    foreground_pixels = np.count_nonzero(
        binary == 255
    )

    total_pixels = binary.size

    return (
        foreground_pixels /
        total_pixels
    ) * 100


def edge_density(edges):
    """
    Percentage of pixels classified as strong edges.
    """

    edge_pixels = np.count_nonzero(
        edges > 50
    )

    total_pixels = edges.size

    return (
        edge_pixels /
        total_pixels
    ) * 100


def sharpness(gray):
    """
    Variance of Laplacian.
    Higher value generally indicates stronger
    high-frequency detail.
    """

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(path, image):
    """
    Save an image and verify successful writing.
    """

    success = cv2.imwrite(
        path,
        image
    )

    if not success:
        raise IOError(
            f"Could not save image: {path}"
        )


# ============================================================
# EDGE COMPARISON FIGURE
# ============================================================

def create_edge_comparison(
    image_name,
    input_image,
    edges
):
    """
    Create a single comparison figure showing
    the input image and all edge detectors.
    """

    input_rgb = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2RGB
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    axes[0, 0].imshow(
        input_rgb
    )
    axes[0, 0].set_title(
        "Best Input from Experiment 5"
    )

    axes[0, 1].imshow(
        edges["Roberts"],
        cmap="gray"
    )
    axes[0, 1].set_title(
        "Roberts"
    )

    axes[0, 2].imshow(
        edges["Prewitt"],
        cmap="gray"
    )
    axes[0, 2].set_title(
        "Prewitt"
    )

    axes[1, 0].imshow(
        edges["Sobel"],
        cmap="gray"
    )
    axes[1, 0].set_title(
        "Sobel"
    )

    axes[1, 1].imshow(
        edges["Laplacian"],
        cmap="gray"
    )
    axes[1, 1].set_title(
        "Laplacian"
    )

    # Hide unused subplot
    axes[1, 2].axis("off")

    for ax in axes.flat:
        ax.axis("off")

    figure.suptitle(
        f"Edge Detection Comparison - {image_name}",
        fontsize=15
    )

    figure.tight_layout()

    output_path = os.path.join(
        COMPARISON_DIR,
        image_name.replace(
            ".jpeg",
            "_edge_comparison.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)


# ============================================================
# THRESHOLD COMPARISON FIGURE
# ============================================================

def create_threshold_comparison(
    image_name,
    input_image,
    global_result,
    otsu_result,
    adaptive_result
):
    """
    Create thresholding comparison figure.
    """

    input_rgb = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2RGB
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(16, 5)
    )

    axes[0].imshow(
        input_rgb
    )
    axes[0].set_title(
        "Best Input"
    )

    axes[1].imshow(
        global_result,
        cmap="gray"
    )
    axes[1].set_title(
        "Global Threshold"
    )

    axes[2].imshow(
        otsu_result,
        cmap="gray"
    )
    axes[2].set_title(
        "Otsu"
    )

    axes[3].imshow(
        adaptive_result,
        cmap="gray"
    )
    axes[3].set_title(
        "Adaptive"
    )

    for ax in axes:
        ax.axis("off")

    figure.suptitle(
        f"Thresholding Comparison - {image_name}",
        fontsize=15
    )

    figure.tight_layout()

    output_path = os.path.join(
        COMPARISON_DIR,
        image_name.replace(
            ".jpeg",
            "_threshold_comparison.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(figure)


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(image_name):
    """
    Complete Experiment 6 processing for one image.
    """

    print(
        f"\nProcessing {image_name}..."
    )

    original = load_image(
        image_name
    )

    # --------------------------------------------------------
    # TASK 1
    # Prepare best input from Experiment 5
    # --------------------------------------------------------

    best_input = prepare_best_input(
        original
    )

    input_path = os.path.join(
        INPUT_DIR,
        image_name
    )

    save_image(
        input_path,
        best_input
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        best_input,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # TASK 2
    # EDGE DETECTION
    # --------------------------------------------------------

    edge_methods = {
        "Roberts": roberts_operator,
        "Prewitt": prewitt_operator,
        "Sobel": sobel_operator,
        "Laplacian": laplacian_operator,
    }

    edge_results = {}

    for method_name, method_function in edge_methods.items():

        start_time = time.perf_counter()

        edges = method_function(
            gray
        )

        execution_time = (
            time.perf_counter() -
            start_time
        )

        edge_results[
            method_name
        ] = edges

        if method_name == "Roberts":
            output_directory = ROBERTS_DIR

        elif method_name == "Prewitt":
            output_directory = PREWITT_DIR

        elif method_name == "Sobel":
            output_directory = SOBEL_DIR

        else:
            output_directory = LAPLACIAN_DIR

        output_path = os.path.join(
            output_directory,
            image_name.replace(
                ".jpeg",
                "_edges.png"
            )
        )

        save_image(
            output_path,
            edges
        )

        print(
            f"{method_name:<12} "
            f"Edge Density: "
            f"{edge_density(edges):.2f}% | "
            f"Time: "
            f"{execution_time:.6f}s"
        )

    create_edge_comparison(
        image_name,
        best_input,
        edge_results
    )

    # --------------------------------------------------------
    # TASK 3
    # THRESHOLDING
    # --------------------------------------------------------

    threshold_results = {}

    # Global threshold
    start_time = time.perf_counter()

    global_result, global_value = (
        global_threshold(gray)
    )

    global_time = (
        time.perf_counter() -
        start_time
    )

    threshold_results[
        "Global Threshold"
    ] = global_result

    # Otsu threshold
    start_time = time.perf_counter()

    otsu_result, otsu_value = (
        otsu_threshold(gray)
    )

    otsu_time = (
        time.perf_counter() -
        start_time
    )

    threshold_results[
        "Otsu"
    ] = otsu_result

    # Adaptive threshold
    start_time = time.perf_counter()

    adaptive_result = (
        adaptive_threshold(gray)
    )

    adaptive_time = (
        time.perf_counter() -
        start_time
    )

    threshold_results[
        "Adaptive Threshold"
    ] = adaptive_result

    # Save threshold results
    save_image(
        os.path.join(
            GLOBAL_DIR,
            image_name.replace(
                ".jpeg",
                "_global.png"
            )
        ),
        global_result
    )

    save_image(
        os.path.join(
            OTSU_DIR,
            image_name.replace(
                ".jpeg",
                "_otsu.png"
            )
        ),
        otsu_result
    )

    save_image(
        os.path.join(
            ADAPTIVE_DIR,
            image_name.replace(
                ".jpeg",
                "_adaptive.png"
            )
        ),
        adaptive_result
    )

    create_threshold_comparison(
        image_name,
        best_input,
        global_result,
        otsu_result,
        adaptive_result
    )

    print(
        f"Global Threshold: "
        f"{global_value:.2f}"
    )

    print(
        f"Otsu Threshold: "
        f"{otsu_value:.2f}"
    )

    print(
        f"Adaptive Threshold: "
        f"block size = 11, C = 2"
    )

    # --------------------------------------------------------
    # CREATE RESULTS
    # --------------------------------------------------------

    results = []

    for method_name, edges in edge_results.items():

        results.append(
            {
                "Image": image_name,
                "Category": "Edge Detection",
                "Method": method_name,
                "Threshold": "",
                "Edge_Density_%": round(
                    edge_density(edges),
                    4
                ),
                "Foreground_%": "",
                "Sharpness": round(
                    sharpness(gray),
                    4
                ),
            }
        )

    threshold_data = [
    (
        "Global Threshold",
        global_result,
        global_value
    ),
    (
        "Otsu",
        otsu_result,
        otsu_value
    ),
    (
        "Adaptive Threshold",
        adaptive_result,
        "Automatic"
    ),
    ]

    for method_name, result, threshold_value in threshold_data:

     results.append(
        {
            "Image": image_name,
            "Category": "Thresholding",
            "Method": method_name,
            "Threshold": threshold_value,
            "Edge_Density_%": "",
            "Foreground_%": round(
                foreground_percentage(result),
                4
            ),
            "Sharpness": round(
                sharpness(gray),
                4
            ),
        }
    ) 

    return results


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

def run_experiment():
    """
    Run Experiment 6 on all three project images.
    """

    all_results = []

    print("\n")
    print("=" * 65)
    print(
        "EXPERIMENT 6 - IMAGE SEGMENTATION"
    )
    print(
        "Edge Detection and Thresholding"
    )
    print("=" * 65)

    for image_name in IMAGE_NAMES:

        try:

            results = process_image(
                image_name
            )

            all_results.extend(
                results
            )

        except Exception as error:

            print(
                f"\nError processing "
                f"{image_name}: {error}"
            )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    dataframe = pd.DataFrame(
        all_results
    )

    dataframe.to_csv(
        RESULTS_CSV,
        index=False
    )

    print("\n")
    print("=" * 65)
    print("EXPERIMENT 6 COMPLETED")
    print("=" * 65)

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        f"\nCSV saved to:"
        f"\n{RESULTS_CSV}"
    )


# ============================================================
# DISPLAY ALL RESULTS FOR ONE IMAGE
# ============================================================

def display_image_results(image_number):

    if image_number < 1 or image_number > 3:

        print(
            "Invalid image selection."
        )

        return

    image_name = IMAGE_NAMES[
        image_number - 1
    ]

    original = load_image(
        image_name
    )

    best_input = prepare_best_input(
        original
    )

    gray = cv2.cvtColor(
        best_input,
        cv2.COLOR_BGR2GRAY
    )

    # Edge detection
    roberts = roberts_operator(
        gray
    )

    prewitt = prewitt_operator(
        gray
    )

    sobel = sobel_operator(
        gray
    )

    laplacian = laplacian_operator(
        gray
    )

    # Thresholding
    global_result, global_value = (
        global_threshold(gray)
    )

    otsu_result, otsu_value = (
        otsu_threshold(gray)
    )

    adaptive_result = (
        adaptive_threshold(gray)
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    input_rgb = cv2.cvtColor(
        best_input,
        cv2.COLOR_BGR2RGB
    )

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(15, 13)
    )

    axes[0, 0].imshow(
        input_rgb
    )
    axes[0, 0].set_title(
        "Best Input - Exp. 5"
    )

    axes[0, 1].imshow(
        roberts,
        cmap="gray"
    )
    axes[0, 1].set_title(
        "Roberts"
    )

    axes[0, 2].imshow(
        prewitt,
        cmap="gray"
    )
    axes[0, 2].set_title(
        "Prewitt"
    )

    axes[1, 0].imshow(
        sobel,
        cmap="gray"
    )
    axes[1, 0].set_title(
        "Sobel"
    )

    axes[1, 1].imshow(
        laplacian,
        cmap="gray"
    )
    axes[1, 1].set_title(
        "Laplacian"
    )

    axes[1, 2].imshow(
        global_result,
        cmap="gray"
    )
    axes[1, 2].set_title(
        f"Global ({global_value:.0f})"
    )

    axes[2, 0].imshow(
        otsu_result,
        cmap="gray"
    )
    axes[2, 0].set_title(
        f"Otsu ({otsu_value:.0f})"
    )

    axes[2, 1].imshow(
        adaptive_result,
        cmap="gray"
    )
    axes[2, 1].set_title(
        "Adaptive"
    )

    axes[2, 2].axis(
        "off"
    )

    for ax in axes.flat:
        ax.axis("off")

    figure.suptitle(
        f"Experiment 6 Segmentation Results - {image_name}",
        fontsize=16
    )

    figure.tight_layout()

    plt.show()


# ============================================================
# INTERACTIVE MENU
# ============================================================

def interactive_menu():

    while True:

        print("\n")
        print("=" * 55)
        print(
            "EXPERIMENT 6 - SEGMENTATION MENU"
        )
        print("=" * 55)

        print(
            "1. Generate all Experiment 6 outputs"
        )

        print(
            "2. View segmentation results"
        )

        print(
            "3. Exit"
        )

        choice = input(
            "\nEnter your choice: "
        ).strip()

        if choice == "1":

            run_experiment()

        elif choice == "2":

            print("\nSelect image:")

            print(
                "1. ss_1.jpeg"
            )

            print(
                "2. ss_2.jpeg"
            )

            print(
                "3. ss_3.jpeg"
            )

            image_choice = input(
                "Enter image number: "
            ).strip()

            try:

                display_image_results(
                    int(image_choice)
                )

            except ValueError:

                print(
                    "Please enter a valid number."
                )

        elif choice == "3":

            print(
                "\nExiting Experiment 6."
            )

            break

        else:

            print(
                "\nInvalid choice."
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    interactive_menu()