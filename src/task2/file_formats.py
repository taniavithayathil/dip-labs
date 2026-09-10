import cv2
import os

# Input image
input_path = "dataset/original/ss_1.jpeg"

# Output directory
output_dir = "outputs/task2/formats"
os.makedirs(output_dir, exist_ok=True)

# Read image
image = cv2.imread(input_path)

# Check if image was loaded
if image is None:
    print("Error: Could not read the image.")
    print(f"Check the file path: {input_path}")
    exit()

# Output file paths
bmp_path = os.path.join(output_dir, "ss_1.bmp")
png_path = os.path.join(output_dir, "ss_1.png")
jpeg_path = os.path.join(output_dir, "ss_1_compressed.jpeg")

# Save in different formats
cv2.imwrite(bmp_path, image)
cv2.imwrite(png_path, image)

# JPEG with quality = 90
cv2.imwrite(
    jpeg_path,
    image,
    [cv2.IMWRITE_JPEG_QUALITY, 90]
)

# Get file sizes in KB
bmp_size = os.path.getsize(bmp_path) / 1024
png_size = os.path.getsize(png_path) / 1024
jpeg_size = os.path.getsize(jpeg_path) / 1024

# Display comparison
print("\n===== IMAGE FILE FORMAT COMPARISON =====")

print(f"\nBMP")
print(f"File: {bmp_path}")
print(f"Size: {bmp_size:.2f} KB")
print("Compression: Uncompressed")

print(f"\nPNG")
print(f"File: {png_path}")
print(f"Size: {png_size:.2f} KB")
print("Compression: Lossless")

print(f"\nJPEG")
print(f"File: {jpeg_path}")
print(f"Size: {jpeg_size:.2f} KB")
print("Compression: Lossy")
print("JPEG quality: 90")

print("\nFile format comparison completed successfully.")
