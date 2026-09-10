import cv2
import os
import matplotlib.pyplot as plt

# Input image
input_path = "dataset/original/ss_1.jpeg"

# Output directory
output_dir = "outputs/task2/color_spaces"
os.makedirs(output_dir, exist_ok=True)

# Read image using OpenCV
image = cv2.imread(input_path)

# Check if image was loaded
if image is None:
    print("Error: Could not read the image.")
    print(f"Check the file path: {input_path}")
    exit()

# Convert BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert BGR to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Save converted images
cv2.imwrite(
    os.path.join(output_dir, "grayscale.png"),
    gray_image
)

cv2.imwrite(
    os.path.join(output_dir, "rgb.png"),
    cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(output_dir, "hsv.png"),
    hsv_image
)

# Display all color spaces
plt.figure(figsize=(15, 8))

# Original RGB
plt.subplot(2, 2, 1)
plt.imshow(rgb_image)
plt.title("Original / RGB")
plt.axis("off")

# Grayscale
plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

# RGB
plt.subplot(2, 2, 3)
plt.imshow(rgb_image)
plt.title("RGB")
plt.axis("off")

# HSV
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
plt.title("HSV")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Color space conversion completed successfully.")
print(f"Outputs saved in: {output_dir}")