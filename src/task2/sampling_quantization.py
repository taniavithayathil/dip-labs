import cv2
import os
import matplotlib.pyplot as plt

# Input image
input_path = "dataset/original/ss_1.jpeg"

# Output directories
sampling_dir = "outputs/task2/sampling"
quantization_dir = "outputs/task2/quantization"

os.makedirs(sampling_dir, exist_ok=True)
os.makedirs(quantization_dir, exist_ok=True)

# Read image
image = cv2.imread(input_path)

# Check if image was loaded
if image is None:
    print("Error: Could not read the image.")
    print(f"Check the file path: {input_path}")
    exit()

# Convert BGR to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ============================================================
# SAMPLING
# ============================================================

# Original dimensions
height, width = image.shape[:2]

# Resize to 100%, 50%, and 25%
sample_100 = image.copy()

sample_50 = cv2.resize(
    image,
    (width // 2, height // 2),
    interpolation=cv2.INTER_AREA
)

sample_25 = cv2.resize(
    image,
    (width // 4, height // 4),
    interpolation=cv2.INTER_AREA
)

# Save sampled images
cv2.imwrite(
    os.path.join(sampling_dir, "sampling_100.png"),
    sample_100
)

cv2.imwrite(
    os.path.join(sampling_dir, "sampling_50.png"),
    sample_50
)

cv2.imwrite(
    os.path.join(sampling_dir, "sampling_25.png"),
    sample_25
)

# Display sampling results
plt.figure(figsize=(12, 10))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(sample_100, cv2.COLOR_BGR2RGB))
plt.title("Sampling - 100%")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(sample_50, cv2.COLOR_BGR2RGB))
plt.title("Sampling - 50%")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(sample_25, cv2.COLOR_BGR2RGB))
plt.title("Sampling - 25%")
plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# QUANTIZATION
# ============================================================

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def quantize_image(image, levels):
    """
    Reduce the number of gray levels in an image.
    """
    step = 256 // levels
    quantized = (image // step) * step

    return quantized.astype("uint8")


# Generate quantized images
quantized_256 = quantize_image(gray, 256)
quantized_128 = quantize_image(gray, 128)
quantized_64 = quantize_image(gray, 64)
quantized_32 = quantize_image(gray, 32)

# Save quantized images
cv2.imwrite(
    os.path.join(quantization_dir, "quantization_256.png"),
    quantized_256
)

cv2.imwrite(
    os.path.join(quantization_dir, "quantization_128.png"),
    quantized_128
)

cv2.imwrite(
    os.path.join(quantization_dir, "quantization_64.png"),
    quantized_64
)

cv2.imwrite(
    os.path.join(quantization_dir, "quantization_32.png"),
    quantized_32
)

# Display quantization results
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(quantized_256, cmap="gray", vmin=0, vmax=255)
plt.title("256 Gray Levels")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(quantized_128, cmap="gray", vmin=0, vmax=255)
plt.title("128 Gray Levels")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(quantized_64, cmap="gray", vmin=0, vmax=255)
plt.title("64 Gray Levels")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(quantized_32, cmap="gray", vmin=0, vmax=255)
plt.title("32 Gray Levels")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nSampling and quantization completed successfully.")

print("\nSampling outputs:")
print(f"100%: {sampling_dir}\\sampling_100.png")
print(f"50% : {sampling_dir}\\sampling_50.png")
print(f"25% : {sampling_dir}\\sampling_25.png")

print("\nQuantization outputs:")
print(f"256 levels: {quantization_dir}\\quantization_256.png")
print(f"128 levels: {quantization_dir}\\quantization_128.png")
print(f"64 levels : {quantization_dir}\\quantization_64.png")
print(f"32 levels : {quantization_dir}\\quantization_32.png")