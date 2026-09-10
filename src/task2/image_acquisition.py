import cv2
import os
import matplotlib.pyplot as plt

# Input image
input_path = "dataset/original/ss_1.jpeg"

# Output directory
output_dir = "outputs/task2/acquisition"
os.makedirs(output_dir, exist_ok=True)


# Read image
image = cv2.imread(input_path)

# Check if image was loaded
if image is None:
    print("Error: Could not read the image.")
    print(f"Check the file path: {input_path}")
    exit()

# Convert BGR to RGB for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(8, 12))
plt.imshow(image_rgb)
plt.title("Original Screenshot")
plt.axis("off")
plt.show()

# Save acquired image
output_path = os.path.join(output_dir, "ss_1.png")
cv2.imwrite(output_path, image)

print("Image acquired successfully.")
print(f"Input : {input_path}")
print(f"Output: {output_path}")