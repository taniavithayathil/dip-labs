"""
Experiment 7 - Region-Based Segmentation

Project:
AI-Assisted Screenshot Authenticity Detection Using Digital Image Processing

Experiment 7 Tasks:
1. Region Growing
2. Compare 4-connected and 8-connected region growing
3. Region Splitting and Merging

Project Pipeline:
Image Acquisition
        ↓
Image Enhancement
        ↓
Filtering
        ↓
Segmentation
        ↓
Region-Based Segmentation
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
    "task7"
)

INPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "input"
)

REGION_GROWING_DIR = os.path.join(
    OUTPUT_DIR,
    "region_growing"
)

CONNECTIVITY_DIR = os.path.join(
    OUTPUT_DIR,
    "connectivity"
)

SPLITTING_DIR = os.path.join(
    OUTPUT_DIR,
    "splitting"
)

MERGING_DIR = os.path.join(
    OUTPUT_DIR,
    "merging"
)

COMPARISON_DIR = os.path.join(
    OUTPUT_DIR,
    "comparisons"
)

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "experiment7_results.csv"
)


# ============================================================
# DATASET
# ============================================================

IMAGE_NAMES = [
    "ss_1.jpeg",
    "ss_2.jpeg",
    "ss_3.jpeg"
]


# ============================================================
# REGION GROWING PARAMETERS
# ============================================================

# At least three threshold values as required.
REGION_THRESHOLDS = [
    10,
    20,
    40
]

# Minimum region size used in splitting.
MIN_REGION_SIZE = 16

# Homogeneity threshold for region splitting.
SPLIT_THRESHOLD = 20

# Similarity threshold for merging adjacent regions.
MERGE_THRESHOLD = 15


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DIRECTORIES = [
    OUTPUT_DIR,
    INPUT_DIR,
    REGION_GROWING_DIR,
    CONNECTIVITY_DIR,
    SPLITTING_DIR,
    MERGING_DIR,
    COMPARISON_DIR
]

for directory in DIRECTORIES:
    os.makedirs(
        directory,
        exist_ok=True
    )

for threshold in REGION_THRESHOLDS:

    os.makedirs(
        os.path.join(
            REGION_GROWING_DIR,
            f"threshold_{threshold}"
        ),
        exist_ok=True
    )

os.makedirs(
    os.path.join(
        CONNECTIVITY_DIR,
        "4_connected"
    ),
    exist_ok=True
)

os.makedirs(
    os.path.join(
        CONNECTIVITY_DIR,
        "8_connected"
    ),
    exist_ok=True
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_name):
    """
    Load an image from the project dataset.
    """

    path = os.path.join(
        DATASET_DIR,
        image_name
    )

    image = cv2.imread(
        path
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    return image


def load_grayscale(image_name):
    """
    Load project image and convert to grayscale.
    """

    image = load_image(
        image_name
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return image, gray


# ============================================================
# SEED SELECTION
# ============================================================

def select_seed(gray):
    """
    Automatically select a seed point.

    The center of the image is used as the default seed.
    """

    height, width = gray.shape

    x = width // 2
    y = height // 2

    return x, y


# ============================================================
# REGION GROWING
# ============================================================

def get_neighbors(
    y,
    x,
    height,
    width,
    connectivity
):
    """
    Return neighboring pixel coordinates.

    4-connected:
        top
        bottom
        left
        right

    8-connected:
        4-connected neighbours
        + diagonal neighbours
    """

    if connectivity == 4:

        offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

    else:

        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1)
        ]

    neighbors = []

    for dy, dx in offsets:

        ny = y + dy
        nx = x + dx

        if (
            0 <= ny < height
            and
            0 <= nx < width
        ):
            neighbors.append(
                (ny, nx)
            )

    return neighbors


def region_growing(
    gray,
    seed,
    threshold,
    connectivity=4
):
    """
    Perform region growing.

    Similarity criterion:

        |pixel_intensity - region_mean| <= threshold

    The region mean is updated whenever a new
    pixel is added.
    """

    height, width = gray.shape

    seed_x, seed_y = seed

    seed_value = float(
        gray[seed_y, seed_x]
    )

    region = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    visited = np.zeros(
        (height, width),
        dtype=bool
    )

    queue = []

    queue.append(
        (seed_y, seed_x)
    )

    visited[
        seed_y,
        seed_x
    ] = True

    region[
        seed_y,
        seed_x
    ] = 255

    region_sum = seed_value
    region_count = 1

    while queue:

        y, x = queue.pop(0)

        region_mean = (
            region_sum /
            region_count
        )

        neighbors = get_neighbors(
            y,
            x,
            height,
            width,
            connectivity
        )

        for ny, nx in neighbors:

            if visited[
                ny,
                nx
            ]:
                continue

            visited[
                ny,
                nx
            ] = True

            pixel_value = float(
                gray[ny, nx]
            )

            difference = abs(
                pixel_value -
                region_mean
            )

            if difference <= threshold:

                region[
                    ny,
                    nx
                ] = 255

                queue.append(
                    (ny, nx)
                )

                region_sum += pixel_value

                region_count += 1

    return region, region_mean, region_count


# ============================================================
# REGION STATISTICS
# ============================================================

def region_percentage(region):
    """
    Calculate percentage of image belonging
    to the segmented region.
    """

    return (
        np.count_nonzero(
            region == 255
        )
        /
        region.size
    ) * 100


# ============================================================
# SPLIT CONDITION
# ============================================================

def is_homogeneous(
    region,
    threshold
):
    """
    Check whether a region is homogeneous.

    Range-based criterion:

        max intensity - min intensity <= threshold
    """

    if region.size == 0:
        return True

    intensity_range = (
        float(np.max(region))
        -
        float(np.min(region))
    )

    return intensity_range <= threshold


# ============================================================
# REGION SPLITTING
# ============================================================

def split_region(
    gray,
    x,
    y,
    width,
    height,
    threshold,
    min_size,
    regions
):
    """
    Recursive quadtree region splitting.

    A region is split into four subregions when
    it does not satisfy the homogeneity criterion.
    """

    region = gray[
        y:y + height,
        x:x + width
    ]

    if (
        width <= min_size
        or
        height <= min_size
        or
        is_homogeneous(
            region,
            threshold
        )
    ):

        regions.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "mean": float(
                    np.mean(region)
                )
            }
        )

        return

    half_width = width // 2
    half_height = height // 2

    if half_width == 0 or half_height == 0:

        regions.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "mean": float(
                    np.mean(region)
                )
            }
        )

        return

    # Top-left
    split_region(
        gray,
        x,
        y,
        half_width,
        half_height,
        threshold,
        min_size,
        regions
    )

    # Top-right
    split_region(
        gray,
        x + half_width,
        y,
        width - half_width,
        half_height,
        threshold,
        min_size,
        regions
    )

    # Bottom-left
    split_region(
        gray,
        x,
        y + half_height,
        half_width,
        height - half_height,
        threshold,
        min_size,
        regions
    )

    # Bottom-right
    split_region(
        gray,
        x + half_width,
        y + half_height,
        width - half_width,
        height - half_height,
        threshold,
        min_size,
        regions
    )


def perform_region_splitting(
    gray,
    threshold=SPLIT_THRESHOLD,
    min_size=MIN_REGION_SIZE
):
    """
    Start region splitting from the complete image.
    """

    height, width = gray.shape

    regions = []

    split_region(
        gray,
        0,
        0,
        width,
        height,
        threshold,
        min_size,
        regions
    )

    return regions


# ============================================================
# VISUALIZE SPLIT REGIONS
# ============================================================

def create_split_visualization(
    gray,
    regions
):
    """
    Draw the resulting split regions.
    """

    visualization = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    for region in regions:

        x = region["x"]
        y = region["y"]

        width = region["width"]
        height = region["height"]

        cv2.rectangle(
            visualization,
            (x, y),
            (
                x + width - 1,
                y + height - 1
            ),
            (255, 255, 255),
            1
        )

    return visualization


# ============================================================
# REGION MERGING
# ============================================================

def regions_are_adjacent(
    region_a,
    region_b
):
    """
    Check whether two rectangular regions
    share a boundary.
    """

    ax1 = region_a["x"]
    ay1 = region_a["y"]

    ax2 = (
        ax1 +
        region_a["width"]
    )

    ay2 = (
        ay1 +
        region_a["height"]
    )

    bx1 = region_b["x"]
    by1 = region_b["y"]

    bx2 = (
        bx1 +
        region_b["width"]
    )

    by2 = (
        by1 +
        region_b["height"]
    )

    horizontal_touch = (
        (
            ax2 == bx1
            or
            bx2 == ax1
        )
        and
        max(ay1, by1)
        <
        min(ay2, by2)
    )

    vertical_touch = (
        (
            ay2 == by1
            or
            by2 == ay1
        )
        and
        max(ax1, bx1)
        <
        min(ax2, bx2)
    )

    return (
        horizontal_touch
        or
        vertical_touch
    )


def merge_two_regions(
    region_a,
    region_b
):
    """
    Merge two adjacent rectangular regions.
    """

    x1 = min(
        region_a["x"],
        region_b["x"]
    )

    y1 = min(
        region_a["y"],
        region_b["y"]
    )

    x2 = max(
        region_a["x"] +
        region_a["width"],
        region_b["x"] +
        region_b["width"]
    )

    y2 = max(
        region_a["y"] +
        region_a["height"],
        region_b["y"] +
        region_b["height"]
    )

    width = x2 - x1
    height = y2 - y1

    area_a = (
        region_a["width"] *
        region_a["height"]
    )

    area_b = (
        region_b["width"] *
        region_b["height"]
    )

    mean = (
        (
            region_a["mean"] *
            area_a
        )
        +
        (
            region_b["mean"] *
            area_b
        )
    ) / (
        area_a +
        area_b
    )

    return {
        "x": x1,
        "y": y1,
        "width": width,
        "height": height,
        "mean": mean
    }


def merge_regions(
    regions,
    threshold=MERGE_THRESHOLD
):
    """
    Iteratively merge adjacent regions whose
    mean intensities are sufficiently similar.
    """

    regions = [
        region.copy()
        for region in regions
    ]

    changed = True

    while changed:

        changed = False

        for i in range(
            len(regions)
        ):

            merged = False

            for j in range(
                i + 1,
                len(regions)
            ):

                region_a = regions[i]
                region_b = regions[j]

                if not regions_are_adjacent(
                    region_a,
                    region_b
                ):
                    continue

                mean_difference = abs(
                    region_a["mean"]
                    -
                    region_b["mean"]
                )

                if (
                    mean_difference
                    <=
                    threshold
                ):

                    merged_region = (
                        merge_two_regions(
                            region_a,
                            region_b
                        )
                    )

                    regions[i] = (
                        merged_region
                    )

                    regions.pop(j)

                    changed = True
                    merged = True

                    break

            if merged:
                break

    return regions


# ============================================================
# VISUALIZE MERGED REGIONS
# ============================================================

def create_merged_visualization(
    gray,
    regions
):
    """
    Create an image representing the
    final merged regions using their
    mean intensities.
    """

    result = np.zeros_like(
        gray
    )

    for region in regions:

        x = region["x"]
        y = region["y"]

        width = region["width"]
        height = region["height"]

        mean_value = int(
            np.clip(
                region["mean"],
                0,
                255
            )
        )

        result[
            y:y + height,
            x:x + width
        ] = mean_value

    return result


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    path,
    image
):
    """
    Save image and verify success.
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
# REGION GROWING COMPARISON FIGURE
# ============================================================

def create_region_growing_comparison(
    image_name,
    gray,
    threshold_results,
    seed
):
    """
    Create comparison of region growing
    for three thresholds.
    """

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(15, 5)
    )

    axes[0].imshow(
        gray,
        cmap="gray"
    )

    axes[0].plot(
        seed[0],
        seed[1],
        "r+",
        markersize=12
    )

    axes[0].set_title(
        "Grayscale + Seed"
    )

    for index, threshold in enumerate(
        REGION_THRESHOLDS
    ):

        axes[index + 1].imshow(
            threshold_results[
                threshold
            ],
            cmap="gray"
        )

        axes[index + 1].set_title(
            f"Threshold = {threshold}"
        )

    for ax in axes:

        ax.axis("off")

    figure.suptitle(
        f"Region Growing - {image_name}",
        fontsize=15
    )

    figure.tight_layout()

    output_path = os.path.join(
        COMPARISON_DIR,
        image_name.replace(
            ".jpeg",
            "_region_growing.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# CONNECTIVITY COMPARISON FIGURE
# ============================================================

def create_connectivity_comparison(
    image_name,
    results
):
    """
    Compare 4-connected and 8-connected
    region growing.
    """

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5)
    )

    axes[0].imshow(
        results[4],
        cmap="gray"
    )

    axes[0].set_title(
        "4-Connected"
    )

    axes[1].imshow(
        results[8],
        cmap="gray"
    )

    axes[1].set_title(
        "8-Connected"
    )

    for ax in axes:

        ax.axis("off")

    figure.suptitle(
        f"Connectivity Comparison - {image_name}",
        fontsize=15
    )

    figure.tight_layout()

    output_path = os.path.join(
        COMPARISON_DIR,
        image_name.replace(
            ".jpeg",
            "_connectivity.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# SPLITTING + MERGING COMPARISON FIGURE
# ============================================================

def create_split_merge_comparison(
    image_name,
    gray,
    split_image,
    merged_image
):
    """
    Compare splitting and merging results.
    """

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        gray,
        cmap="gray"
    )

    axes[0].set_title(
        "Original Grayscale"
    )

    axes[1].imshow(
        split_image,
        cmap="gray"
    )

    axes[1].set_title(
        "Region Splitting"
    )

    axes[2].imshow(
        merged_image,
        cmap="gray"
    )

    axes[2].set_title(
        "Region Merging"
    )

    for ax in axes:

        ax.axis("off")

    figure.suptitle(
        f"Region Splitting and Merging - {image_name}",
        fontsize=15
    )

    figure.tight_layout()

    output_path = os.path.join(
        COMPARISON_DIR,
        image_name.replace(
            ".jpeg",
            "_split_merge.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_name
):
    """
    Complete Experiment 7 processing
    for one image.
    """

    print(
        f"\nProcessing {image_name}..."
    )

    original, gray = load_grayscale(
        image_name
    )

    height, width = gray.shape

    minimum_intensity = int(
        np.min(gray)
    )

    maximum_intensity = int(
        np.max(gray)
    )

    # --------------------------------------------------------
    # TASK 1 - INPUT PREPARATION
    # --------------------------------------------------------

    input_path = os.path.join(
        INPUT_DIR,
        image_name.replace(
            ".jpeg",
            "_grayscale.png"
        )
    )

    save_image(
        input_path,
        gray
    )

    # Automatic seed at image centre
    seed = select_seed(
        gray
    )

    seed_x, seed_y = seed

    seed_value = int(
        gray[seed_y, seed_x]
    )

    print(
        f"Dimensions: "
        f"{width} x {height}"
    )

    print(
        f"Intensity range: "
        f"{minimum_intensity} - "
        f"{maximum_intensity}"
    )

    print(
        f"Seed point: "
        f"({seed_x}, {seed_y})"
    )

    print(
        f"Seed intensity: "
        f"{seed_value}"
    )

    all_results = []

    # --------------------------------------------------------
    # TASK 1 - REGION GROWING
    # --------------------------------------------------------

    threshold_results = {}

    for threshold in REGION_THRESHOLDS:

        start_time = time.perf_counter()

        region, region_mean, region_count = (
            region_growing(
                gray,
                seed,
                threshold,
                connectivity=8
            )
        )

        processing_time = (
            time.perf_counter()
            -
            start_time
        )

        threshold_results[
            threshold
        ] = region

        output_path = os.path.join(
            REGION_GROWING_DIR,
            f"threshold_{threshold}",
            image_name.replace(
                ".jpeg",
                "_region.png"
            )
        )

        save_image(
            output_path,
            region
        )

        percentage = region_percentage(
            region
        )

        print(
            f"Region Growing | "
            f"T={threshold} | "
            f"Pixels={region_count} | "
            f"Region={percentage:.2f}% | "
            f"Time={processing_time:.6f}s"
        )

        all_results.append(
            {
                "Image": image_name,
                "Task": "Region Growing",
                "Method": "8-Connected",
                "Parameter": threshold,
                "Region_Count": region_count,
                "Region_Percentage": round(
                    percentage,
                    4
                ),
                "Processing_Time": round(
                    processing_time,
                    6
                ),
                "Region_Count_After_Merging": ""
            }
        )

    create_region_growing_comparison(
        image_name,
        gray,
        threshold_results,
        seed
    )

    # --------------------------------------------------------
    # TASK 2 - 4 vs 8 CONNECTIVITY
    # --------------------------------------------------------

    connectivity_results = {}

    connectivity_threshold = 20

    for connectivity in [4, 8]:

        start_time = time.perf_counter()

        region, region_mean, region_count = (
            region_growing(
                gray,
                seed,
                connectivity_threshold,
                connectivity
            )
        )

        processing_time = (
            time.perf_counter()
            -
            start_time
        )

        connectivity_results[
            connectivity
        ] = region

        output_directory = os.path.join(
            CONNECTIVITY_DIR,
            f"{connectivity}_connected"
        )

        output_path = os.path.join(
            output_directory,
            image_name.replace(
                ".jpeg",
                "_region.png"
            )
        )

        save_image(
            output_path,
            region
        )

        percentage = region_percentage(
            region
        )

        print(
            f"{connectivity}-Connected | "
            f"Pixels={region_count} | "
            f"Region={percentage:.2f}% | "
            f"Time={processing_time:.6f}s"
        )

        all_results.append(
            {
                "Image": image_name,
                "Task": "Connectivity Comparison",
                "Method": f"{connectivity}-Connected",
                "Parameter": connectivity_threshold,
                "Region_Count": region_count,
                "Region_Percentage": round(
                    percentage,
                    4
                ),
                "Processing_Time": round(
                    processing_time,
                    6
                ),
                "Region_Count_After_Merging": ""
            }
        )

    create_connectivity_comparison(
        image_name,
        connectivity_results
    )

    # --------------------------------------------------------
    # TASK 3 - REGION SPLITTING
    # --------------------------------------------------------

    start_time = time.perf_counter()

    split_regions = (
        perform_region_splitting(
            gray,
            SPLIT_THRESHOLD,
            MIN_REGION_SIZE
        )
    )

    split_time = (
        time.perf_counter()
        -
        start_time
    )

    split_visualization = (
        create_split_visualization(
            gray,
            split_regions
        )
    )

    split_path = os.path.join(
        SPLITTING_DIR,
        image_name.replace(
            ".jpeg",
            "_split.png"
        )
    )

    save_image(
        split_path,
        split_visualization
    )

    print(
        f"Region Splitting | "
        f"Regions={len(split_regions)} | "
        f"Time={split_time:.6f}s"
    )

    all_results.append(
        {
            "Image": image_name,
            "Task": "Region Splitting",
            "Method": "Quadtree",
            "Parameter": SPLIT_THRESHOLD,
            "Region_Count": len(
                split_regions
            ),
            "Region_Percentage": "",
            "Processing_Time": round(
                split_time,
                6
            ),
            "Region_Count_After_Merging": ""
        }
    )

    # --------------------------------------------------------
    # TASK 3 - REGION MERGING
    # --------------------------------------------------------

    start_time = time.perf_counter()

    merged_regions = merge_regions(
        split_regions,
        MERGE_THRESHOLD
    )

    merge_time = (
        time.perf_counter()
        -
        start_time
    )

    merged_visualization = (
        create_merged_visualization(
            gray,
            merged_regions
        )
    )

    merge_path = os.path.join(
        MERGING_DIR,
        image_name.replace(
            ".jpeg",
            "_merged.png"
        )
    )

    save_image(
        merge_path,
        merged_visualization
    )

    print(
        f"Region Merging | "
        f"Before={len(split_regions)} | "
        f"After={len(merged_regions)} | "
        f"Time={merge_time:.6f}s"
    )

    all_results.append(
        {
            "Image": image_name,
            "Task": "Region Merging",
            "Method": "Adjacent Mean Similarity",
            "Parameter": MERGE_THRESHOLD,
            "Region_Count": len(
                split_regions
            ),
            "Region_Percentage": "",
            "Processing_Time": round(
                merge_time,
                6
            ),
            "Region_Count_After_Merging": len(
                merged_regions
            )
        }
    )

    create_split_merge_comparison(
        image_name,
        gray,
        split_visualization,
        merged_visualization
    )

    return all_results


# ============================================================
# RUN COMPLETE EXPERIMENT
# ============================================================

def run_experiment():

    print("\n")
    print("=" * 70)
    print(
        "EXPERIMENT 7 - REGION-BASED SEGMENTATION"
    )
    print("=" * 70)

    print(
        "\nRegion Growing thresholds:"
    )

    print(
        REGION_THRESHOLDS
    )

    print(
        f"\nSplit threshold: "
        f"{SPLIT_THRESHOLD}"
    )

    print(
        f"Minimum region size: "
        f"{MIN_REGION_SIZE}"
    )

    print(
        f"Merge threshold: "
        f"{MERGE_THRESHOLD}"
    )

    all_results = []

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
    print("=" * 70)
    print(
        "EXPERIMENT 7 COMPLETED"
    )
    print("=" * 70)

    print(
        f"\nOutputs:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        f"\nResults CSV:"
        f"\n{RESULTS_CSV}"
    )


# ============================================================
# DISPLAY ONE IMAGE
# ============================================================

def display_results(
    image_number
):

    if (
        image_number < 1
        or
        image_number > 3
    ):

        print(
            "Invalid image number."
        )

        return

    image_name = IMAGE_NAMES[
        image_number - 1
    ]

    _, gray = load_grayscale(
        image_name
    )

    seed = select_seed(
        gray
    )

    # Region growing
    region_results = {}

    for threshold in REGION_THRESHOLDS:

        region, _, _ = region_growing(
            gray,
            seed,
            threshold,
            connectivity=8
        )

        region_results[
            threshold
        ] = region

    # Connectivity
    region_4, _, _ = region_growing(
        gray,
        seed,
        20,
        4
    )

    region_8, _, _ = region_growing(
        gray,
        seed,
        20,
        8
    )

    # Splitting
    split_regions = (
        perform_region_splitting(
            gray
        )
    )

    split_visualization = (
        create_split_visualization(
            gray,
            split_regions
        )
    )

    # Merging
    merged_regions = merge_regions(
        split_regions
    )

    merged_visualization = (
        create_merged_visualization(
            gray,
            merged_regions
        )
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(15, 14)
    )

    axes[0, 0].imshow(
        gray,
        cmap="gray"
    )

    axes[0, 0].plot(
        seed[0],
        seed[1],
        "r+",
        markersize=12
    )

    axes[0, 0].set_title(
        "Grayscale + Seed"
    )

    axes[0, 1].imshow(
        region_results[10],
        cmap="gray"
    )

    axes[0, 1].set_title(
        "Region Growing T=10"
    )

    axes[0, 2].imshow(
        region_results[20],
        cmap="gray"
    )

    axes[0, 2].set_title(
        "Region Growing T=20"
    )

    axes[1, 0].imshow(
        region_results[40],
        cmap="gray"
    )

    axes[1, 0].set_title(
        "Region Growing T=40"
    )

    axes[1, 1].imshow(
        region_4,
        cmap="gray"
    )

    axes[1, 1].set_title(
        "4-Connected T=20"
    )

    axes[1, 2].imshow(
        region_8,
        cmap="gray"
    )

    axes[1, 2].set_title(
        "8-Connected T=20"
    )

    axes[2, 0].imshow(
        split_visualization,
        cmap="gray"
    )

    axes[2, 0].set_title(
        "Region Splitting"
    )

    axes[2, 1].imshow(
        merged_visualization,
        cmap="gray"
    )

    axes[2, 1].set_title(
        "Region Merging"
    )

    axes[2, 2].axis(
        "off"
    )

    for ax in axes.flat:
        ax.axis("off")

    figure.suptitle(
        f"Experiment 7 Results - {image_name}",
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
        print("=" * 60)
        print(
            "EXPERIMENT 7 - REGION SEGMENTATION MENU"
        )
        print("=" * 60)

        print(
            "1. Generate all Experiment 7 outputs"
        )

        print(
            "2. View results for an image"
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

                display_results(
                    int(image_choice)
                )

            except ValueError:

                print(
                    "Please enter a valid number."
                )

        elif choice == "3":

            print(
                "\nExiting Experiment 7."
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