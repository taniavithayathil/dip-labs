import cv2

# Input image
input_path = "dataset/original/ss_1.jpeg"

# Read the image
image = cv2.imread(input_path)

# Check if image was loaded
if image is None:
    print("Error: Could not read the image.")
    print(f"Check the file path: {input_path}")
    exit()

# Get image dimensions
height, width = image.shape[:2]

# Get number of channels
channels = image.shape[2] if len(image.shape) == 3 else 1

# Calculate total number of pixels
total_pixels = height * width

# Get data type
data_type = image.dtype

# Display image information
print("\n===== IMAGE REPRESENTATION =====")
print(f"Image dimensions : {height} x {width}")
print(f"Number of channels: {channels}")
print(f"Total pixels     : {total_pixels}")
print(f"Data type        : {data_type}")

# Ask user for pixel coordinate
print("\nEnter a pixel coordinate.")
print(f"Valid X range: 0 to {width - 1}")
print(f"Valid Y range: 0 to {height - 1}")

x = int(input("Enter X coordinate: "))
y = int(input("Enter Y coordinate: "))

# Check coordinate validity
if 0 <= x < width and 0 <= y < height:

    # OpenCV stores pixels as BGR
    pixel_value = image[y, x]

    print("\n===== PIXEL INFORMATION =====")
    print(f"Coordinate (X, Y): ({x}, {y})")
    print(f"Pixel value (B, G, R): {pixel_value}")

else:
    print("\nError: Coordinate is outside the image.")