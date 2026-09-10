import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset" / "original"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "task3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """
    Load an image using OpenCV.
    """

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image:\n{image_path}"
        )

    return image


# ============================================================
# DISPLAY HELPER
# ============================================================

def display_image(image, title="Image"):
    """
    Display a BGR image correctly using Matplotlib.
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(7, 8))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# 1. IMAGE NEGATIVE
# ============================================================

def image_negative(image):
    """
    Image Negative Transformation

    Formula:
        s = (L - 1) - r

    For an 8-bit image:
        s = 255 - r
    """

    return 255 - image


# ============================================================
# 2. LOG TRANSFORMATION
# ============================================================

def log_transform(image):
    """
    Logarithmic Transformation

    Formula:
        s = c * log(1 + r)
    """

    image_float = image.astype(np.float32)

    max_value = np.max(image_float)

    if max_value == 0:
        return image.copy()

    c = 255 / np.log(1 + max_value)

    log_image = c * np.log(1 + image_float)

    log_image = np.clip(log_image, 0, 255)

    return np.uint8(log_image)


# ============================================================
# 3. GAMMA / POWER-LAW TRANSFORMATION
# ============================================================

def gamma_transform(image, gamma):
    """
    Gamma / Power-Law Transformation

    Formula:
        s = c * r^gamma

    gamma < 1  -> brighter
    gamma = 1  -> approximately unchanged
    gamma > 1  -> darker
    """

    if gamma <= 0:
        raise ValueError("Gamma must be greater than 0.")

    normalized = image.astype(np.float32) / 255.0

    gamma_image = np.power(normalized, gamma)

    gamma_image = gamma_image * 255.0

    gamma_image = np.clip(gamma_image, 0, 255)

    return np.uint8(gamma_image)


# ============================================================
# 4. CONTRAST STRETCHING
# ============================================================

def contrast_stretch(image):
    """
    Contrast Stretching

    Maps minimum intensity to 0
    and maximum intensity to 255.
    """

    image_float = image.astype(np.float32)

    min_value = np.min(image_float)
    max_value = np.max(image_float)

    if max_value == min_value:
        return image.copy()

    stretched = (
        (image_float - min_value)
        / (max_value - min_value)
        * 255.0
    )

    stretched = np.clip(stretched, 0, 255)

    return np.uint8(stretched)


# ============================================================
# 5. GRAYSCALE CONVERSION
# ============================================================

def to_grayscale(image):
    """
    Convert BGR image to grayscale.
    """

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ============================================================
# 6. HISTOGRAM GENERATION
# ============================================================

def calculate_histogram(image):
    """
    Calculate grayscale histogram.

    Returns:
        Histogram containing 256 intensity bins.
    """

    gray = to_grayscale(image)

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    return histogram


# ============================================================
# HISTOGRAM DISPLAY
# ============================================================

def show_histogram(image, title="Histogram"):
    """
    Display grayscale histogram of an image.
    """

    histogram = calculate_histogram(image)

    plt.figure(figsize=(8, 5))

    plt.plot(histogram)

    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Number of Pixels")
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()


# ============================================================
# IMAGE + HISTOGRAM
# ============================================================

def show_image_and_histogram(image, title="Image"):
    """
    Display image and its grayscale histogram side by side.
    """

    gray = to_grayscale(image)
    histogram = calculate_histogram(image)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 5))

    # Image
    plt.subplot(1, 2, 1)

    plt.imshow(image_rgb)

    plt.title(title)

    plt.axis("off")

    # Histogram
    plt.subplot(1, 2, 2)

    plt.plot(histogram)

    plt.title(f"{title} - Histogram")

    plt.xlabel("Intensity")
    plt.ylabel("Number of Pixels")

    plt.xlim([0, 256])

    plt.tight_layout()

    plt.show()


# ============================================================
# HISTOGRAM COMPARISON
# ============================================================

def compare_histograms(
    original,
    enhanced,
    original_title="Original",
    enhanced_title="Enhanced"
):
    """
    Compare histograms before and after enhancement.
    """

    original_gray = to_grayscale(original)
    enhanced_gray = to_grayscale(enhanced)

    original_hist = cv2.calcHist(
        [original_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    enhanced_hist = cv2.calcHist(
        [enhanced_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        original_hist,
        label=original_title
    )

    plt.plot(
        enhanced_hist,
        label=enhanced_title
    )

    plt.title("Histogram Comparison")

    plt.xlabel("Intensity")

    plt.ylabel("Number of Pixels")

    plt.xlim([0, 256])

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# IMAGE + HISTOGRAM COMPARISON
# ============================================================

def compare_image_and_histogram(
    original,
    enhanced,
    enhanced_name="Enhanced Image"
):
    """
    Display original and enhanced images along with
    their histograms.
    """

    original_gray = to_grayscale(original)
    enhanced_gray = to_grayscale(enhanced)

    original_hist = cv2.calcHist(
        [original_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    enhanced_hist = cv2.calcHist(
        [enhanced_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    enhanced_rgb = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(14, 8))

    # Original image
    plt.subplot(2, 2, 1)

    plt.imshow(original_rgb)

    plt.title("Original Image")

    plt.axis("off")

    # Enhanced image
    plt.subplot(2, 2, 2)

    plt.imshow(enhanced_rgb)

    plt.title(enhanced_name)

    plt.axis("off")

    # Original histogram
    plt.subplot(2, 2, 3)

    plt.plot(original_hist)

    plt.title("Original Histogram")

    plt.xlabel("Intensity")

    plt.ylabel("Number of Pixels")

    plt.xlim([0, 256])

    # Enhanced histogram
    plt.subplot(2, 2, 4)

    plt.plot(enhanced_hist)

    plt.title(f"{enhanced_name} Histogram")

    plt.xlabel("Intensity")

    plt.ylabel("Number of Pixels")

    plt.xlim([0, 256])

    plt.tight_layout()

    plt.show()


# ============================================================
# 7. HISTOGRAM EQUALIZATION
# ============================================================

def histogram_equalization(image):
    """
    Histogram Equalization.

    Equalization is performed on the grayscale image
    because OpenCV's equalizeHist operates on a single
    intensity channel.
    """

    gray = to_grayscale(image)

    equalized = cv2.equalizeHist(gray)

    return equalized


# ============================================================
# DISPLAY HISTOGRAM EQUALIZATION
# ============================================================

def show_histogram_equalization(image):
    """
    Display original image, equalized image,
    and histograms of both.
    """

    gray = to_grayscale(image)

    equalized = histogram_equalization(image)

    original_hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    equalized_hist = cv2.calcHist(
        [equalized],
        [0],
        None,
        [256],
        [0, 256]
    )

    plt.figure(figsize=(12, 8))

    # Original
    plt.subplot(2, 2, 1)

    plt.imshow(gray, cmap="gray")

    plt.title("Original Grayscale Image")

    plt.axis("off")

    # Equalized
    plt.subplot(2, 2, 2)

    plt.imshow(equalized, cmap="gray")

    plt.title("Histogram Equalized Image")

    plt.axis("off")

    # Original histogram
    plt.subplot(2, 2, 3)

    plt.plot(original_hist)

    plt.title("Original Histogram")

    plt.xlabel("Intensity")

    plt.ylabel("Number of Pixels")

    plt.xlim([0, 256])

    # Equalized histogram
    plt.subplot(2, 2, 4)

    plt.plot(equalized_hist)

    plt.title("Equalized Histogram")

    plt.xlabel("Intensity")

    plt.ylabel("Number of Pixels")

    plt.xlim([0, 256])

    plt.tight_layout()

    plt.show()

    return equalized


# ============================================================
# 8. IMAGE ADDITION
# ============================================================

def image_addition(image1, image2):
    """
    Image Addition.

    OpenCV saturation arithmetic is used to prevent
    uint8 overflow.
    """

    image1, image2 = prepare_same_size(
        image1,
        image2
    )

    return cv2.add(image1, image2)


# ============================================================
# 9. IMAGE SUBTRACTION
# ============================================================

def image_subtraction(image1, image2):
    """
    Image Subtraction.

    Computes image1 - image2 using saturated subtraction.
    """

    image1, image2 = prepare_same_size(
        image1,
        image2
    )

    return cv2.subtract(image1, image2)


# ============================================================
# 10. IMAGE AVERAGING
# ============================================================

def image_averaging(image1, image2):
    """
    Image Averaging.

    Computes:
        (image1 + image2) / 2
    """

    image1, image2 = prepare_same_size(
        image1,
        image2
    )

    return cv2.addWeighted(
        image1,
        0.5,
        image2,
        0.5,
        0
    )


# ============================================================
# PREPARE TWO IMAGES FOR ARITHMETIC
# ============================================================

def prepare_same_size(image1, image2):
    """
    Resize second image to match the first image
    if their dimensions are different.
    """

    if image1.shape[:2] != image2.shape[:2]:

        image2 = cv2.resize(
            image2,
            (
                image1.shape[1],
                image1.shape[0]
            )
        )

    return image1, image2


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(image, filename):
    """
    Save an image inside outputs/task3.
    """

    output_path = OUTPUT_DIR / filename

    cv2.imwrite(
        str(output_path),
        image
    )

    print(f"\nSaved: {output_path}")


# ============================================================
# 11. APPLY ALL ENHANCEMENTS TO ONE IMAGE
# ============================================================

def apply_all_enhancements(image, gamma):
    """
    Apply all required intensity enhancement techniques.
    """

    results = {}

    results["Negative"] = image_negative(image)

    results["Log Transformation"] = log_transform(image)

    results["Gamma Transformation"] = gamma_transform(
        image,
        gamma
    )

    results["Contrast Stretching"] = contrast_stretch(
        image
    )

    return results


# ============================================================
# DISPLAY ALL ENHANCEMENTS
# ============================================================

def display_all_enhancements(image, gamma):
    """
    Display original image and all four intensity
    transformations.
    """

    results = apply_all_enhancements(
        image,
        gamma
    )

    plt.figure(figsize=(16, 10))

    # Original
    plt.subplot(2, 3, 1)

    plt.imshow(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Original Image")

    plt.axis("off")

    # Negative
    plt.subplot(2, 3, 2)

    plt.imshow(
        cv2.cvtColor(
            results["Negative"],
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Image Negative")

    plt.axis("off")

    # Log
    plt.subplot(2, 3, 3)

    plt.imshow(
        cv2.cvtColor(
            results["Log Transformation"],
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Log Transformation")

    plt.axis("off")

    # Gamma
    plt.subplot(2, 3, 4)

    plt.imshow(
        cv2.cvtColor(
            results["Gamma Transformation"],
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        f"Gamma Transformation (γ = {gamma})"
    )

    plt.axis("off")

    # Contrast Stretching
    plt.subplot(2, 3, 5)

    plt.imshow(
        cv2.cvtColor(
            results["Contrast Stretching"],
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title("Contrast Stretching")

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# 12. THREE-IMAGE COMPARISON
# ============================================================

def compare_three_images(paths, gamma):
    """
    Apply all enhancement techniques to three images.

    This is used for Task 6.
    """

    images = []

    for path in paths:
        images.append(load_image(path))

    technique_functions = {
        "Negative": image_negative,
        "Log": log_transform,
        "Gamma": lambda img: gamma_transform(
            img,
            gamma
        ),
        "Contrast Stretching": contrast_stretch
    }

    for index, image in enumerate(images, start=1):

        print(
            f"\nProcessing Image {index}: "
            f"{paths[index - 1]}"
        )

        plt.figure(figsize=(16, 8))

        # Original
        plt.subplot(2, 3, 1)

        plt.imshow(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )
        )

        plt.title(
            f"Image {index} - Original"
        )

        plt.axis("off")

        subplot_position = 2

        for name, function in technique_functions.items():

            enhanced = function(image)

            plt.subplot(
                2,
                3,
                subplot_position
            )

            plt.imshow(
                cv2.cvtColor(
                    enhanced,
                    cv2.COLOR_BGR2RGB
                )
            )

            if name == "Gamma":
                title = f"Gamma (γ={gamma})"
            else:
                title = name

            plt.title(title)

            plt.axis("off")

            subplot_position += 1

        plt.tight_layout()

        plt.show()


# ============================================================
# 13. SAVE ALL ENHANCEMENTS
# ============================================================

def save_all_enhancements(image, gamma):
    """
    Save all four intensity transformations.
    """

    results = apply_all_enhancements(
        image,
        gamma
    )

    save_image(
        results["Negative"],
        "negative.png"
    )

    save_image(
        results["Log Transformation"],
        "log.png"
    )

    save_image(
        results["Gamma Transformation"],
        "gamma.png"
    )

    save_image(
        results["Contrast Stretching"],
        "contrast_stretching.png"
    )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    # Default project image
    current_image_path = DATASET_DIR / "ss_1.jpeg"

    current_image = load_image(
        current_image_path
    )

    print("\n" + "=" * 60)
    print("       DIGITAL IMAGE PROCESSING")
    print("          IMAGE ENHANCEMENT MODULE")
    print("=" * 60)

    print(
        f"\nCurrent image:\n{current_image_path}"
    )

    while True:

        print("\n" + "-" * 60)

        print("IMAGE ENHANCEMENT MENU")

        print("-" * 60)

        print("1.  Load Image")
        print("2.  Image Negative")
        print("3.  Log Transformation")
        print("4.  Gamma Transformation")
        print("5.  Contrast Stretching")
        print("6.  Show Original Histogram")
        print("7.  Compare Original and Enhanced Histogram")
        print("8.  Histogram Equalization")
        print("9.  Image Addition")
        print("10. Image Subtraction")
        print("11. Image Averaging")
        print("12. Apply All Enhancements")
        print("13. Three-Image Comparison")
        print("14. Save All Enhancements")
        print("0.  Exit")

        print("-" * 60)

        choice = input(
            "Enter your choice: "
        ).strip()

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        if choice == "1":

            path = input(
                "\nEnter image path: "
            ).strip()

            try:

                current_image = load_image(path)

                current_image_path = Path(path)

                print(
                    "\nImage loaded successfully."
                )

                print(
                    f"Size: "
                    f"{current_image.shape[1]} x "
                    f"{current_image.shape[0]}"
                )

            except FileNotFoundError as error:

                print(f"\nERROR: {error}")

        # ====================================================
        # NEGATIVE
        # ====================================================

        elif choice == "2":

            result = image_negative(
                current_image
            )

            display_image(
                result,
                "Image Negative"
            )

            save_image(
                result,
                "negative.png"
            )

        # ====================================================
        # LOG
        # ====================================================

        elif choice == "3":

            result = log_transform(
                current_image
            )

            compare_image_and_histogram(
                current_image,
                result,
                "Log Transformation"
            )

            save_image(
                result,
                "log.png"
            )

        # ====================================================
        # GAMMA
        # ====================================================

        elif choice == "4":

            try:

                gamma = float(
                    input(
                        "\nEnter gamma value "
                        "(e.g. 0.5, 1.0, 2.0): "
                    )
                )

                result = gamma_transform(
                    current_image,
                    gamma
                )

                compare_image_and_histogram(
                    current_image,
                    result,
                    f"Gamma Transformation (γ={gamma})"
                )

                save_image(
                    result,
                    f"gamma_{gamma}.png"
                )

            except ValueError as error:

                print(
                    f"\nERROR: {error}"
                )

        # ====================================================
        # CONTRAST STRETCHING
        # ====================================================

        elif choice == "5":

            result = contrast_stretch(
                current_image
            )

            compare_image_and_histogram(
                current_image,
                result,
                "Contrast Stretching"
            )

            save_image(
                result,
                "contrast_stretching.png"
            )

        # ====================================================
        # ORIGINAL HISTOGRAM
        # ====================================================

        elif choice == "6":

            show_image_and_histogram(
                current_image,
                "Original Image"
            )

        # ====================================================
        # HISTOGRAM COMPARISON
        # ====================================================

        elif choice == "7":

            print("\nSelect enhancement:")

            print("1. Negative")
            print("2. Log Transformation")
            print("3. Gamma Transformation")
            print("4. Contrast Stretching")

            enhancement_choice = input(
                "Enter choice: "
            ).strip()

            if enhancement_choice == "1":

                result = image_negative(
                    current_image
                )

                name = "Negative"

            elif enhancement_choice == "2":

                result = log_transform(
                    current_image
                )

                name = "Log Transformation"

            elif enhancement_choice == "3":

                try:

                    gamma = float(
                        input(
                            "Enter gamma value: "
                        )
                    )

                    result = gamma_transform(
                        current_image,
                        gamma
                    )

                    name = (
                        f"Gamma Transformation "
                        f"(γ={gamma})"
                    )

                except ValueError:

                    print(
                        "\nInvalid gamma value."
                    )

                    continue

            elif enhancement_choice == "4":

                result = contrast_stretch(
                    current_image
                )

                name = "Contrast Stretching"

            else:

                print(
                    "\nInvalid choice."
                )

                continue

            compare_image_and_histogram(
                current_image,
                result,
                name
            )

        # ====================================================
        # HISTOGRAM EQUALIZATION
        # ====================================================

        elif choice == "8":

            equalized = (
                show_histogram_equalization(
                    current_image
                )
            )

            save_image(
                equalized,
                "histogram_equalized.png"
            )

        # ====================================================
        # IMAGE ADDITION
        # ====================================================

        elif choice == "9":

            path = input(
                "\nEnter second image path: "
            ).strip()

            try:

                image2 = load_image(path)

                result = image_addition(
                    current_image,
                    image2
                )

                display_image(
                    result,
                    "Image Addition"
                )

                save_image(
                    result,
                    "addition.png"
                )

            except FileNotFoundError as error:

                print(
                    f"\nERROR: {error}"
                )

        # ====================================================
        # IMAGE SUBTRACTION
        # ====================================================

        elif choice == "10":

            path = input(
                "\nEnter second image path: "
            ).strip()

            try:

                image2 = load_image(path)

                result = image_subtraction(
                    current_image,
                    image2
                )

                display_image(
                    result,
                    "Image Subtraction"
                )

                save_image(
                    result,
                    "subtraction.png"
                )

            except FileNotFoundError as error:

                print(
                    f"\nERROR: {error}"
                )

        # ====================================================
        # IMAGE AVERAGING
        # ====================================================

        elif choice == "11":

            path = input(
                "\nEnter second image path: "
            ).strip()

            try:

                image2 = load_image(path)

                result = image_averaging(
                    current_image,
                    image2
                )

                display_image(
                    result,
                    "Image Averaging"
                )

                save_image(
                    result,
                    "averaging.png"
                )

            except FileNotFoundError as error:

                print(
                    f"\nERROR: {error}"
                )

        # ====================================================
        # ALL ENHANCEMENTS
        # ====================================================

        elif choice == "12":

            try:

                gamma = float(
                    input(
                        "\nEnter gamma value "
                        "(recommended: 0.5): "
                    )
                )

                display_all_enhancements(
                    current_image,
                    gamma
                )

            except ValueError as error:

                print(
                    f"\nERROR: {error}"
                )

        # ====================================================
        # THREE IMAGE COMPARISON
        # ====================================================

        elif choice == "13":

            print(
                "\nEnter paths for three different "
                "dataset images."
            )

            path1 = input(
                "Image 1 path: "
            ).strip()

            path2 = input(
                "Image 2 path: "
            ).strip()

            path3 = input(
                "Image 3 path: "
            ).strip()

            try:

                gamma = float(
                    input(
                        "Gamma value "
                        "(recommended: 0.5): "
                    )
                )

                compare_three_images(
                    [
                        path1,
                        path2,
                        path3
                    ],
                    gamma
                )

            except FileNotFoundError as error:

                print(
                    f"\nERROR: {error}"
                )

            except ValueError as error:

                print(
                    f"\nERROR: {error}"
                )

        # ====================================================
        # SAVE ALL
        # ====================================================

        elif choice == "14":

            try:

                gamma = float(
                    input(
                        "\nEnter gamma value "
                        "(recommended: 0.5): "
                    )
                )

                save_all_enhancements(
                    current_image,
                    gamma
                )

                print(
                    "\nAll enhancement results "
                    "saved successfully."
                )

            except ValueError as error:

                print(
                    f"\nERROR: {error}"
                )

        # ====================================================
        # EXIT
        # ====================================================

        elif choice == "0":

            print(
                "\nExiting Image Enhancement Module."
            )

            break

        # ====================================================
        # INVALID CHOICE
        # ====================================================

        else:

            print(
                "\nInvalid choice. "
                "Please select a valid option."
            )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()