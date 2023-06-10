import os
import cv2

def binarize_images_with_otsu(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Construct the full paths to the input and output images
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            # Load the image in grayscale
            image = cv2.imread(input_image_path, 0)

            # Apply Otsu's thresholding
            _, binary_image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Save the binarized image
            cv2.imwrite(output_image_path, binary_image)

            print(f"Binarized image saved: {output_image_path}")

# Provide the paths to the input and output folders
input_folder = "E:\\Pill Project\\rembg_test_data\\rexo"
output_folder = "E:\\Pill Project\\binary_data\\a"

# Binarize images and save them to the output folder
binarize_images_with_otsu(input_folder, output_folder)
