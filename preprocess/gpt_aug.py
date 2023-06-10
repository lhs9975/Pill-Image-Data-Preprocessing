import imgaug.augmenters as iaa
import cv2
import os

# Directory containing the input pill images
input_dir = "E:\\Pill Project\\resize_data\\ace\\b"

# Directory where the augmented images will be saved
output_dir = "E:\\Pill Project\\resize_data\\a"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the augmentation sequence
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),  # Randomly flip images horizontally
    iaa.Affine(rotate=(-10, 10)),  # Randomly rotate images within the range (-10, 10) degrees
    iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply random Gaussian blur
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # Add random Gaussian noise
    iaa.Multiply((0.8, 1.2))  # Multiply pixel values by random factors within the range (0.8, 1.2)
])

# Iterate over all the files in the input directory
for filename in os.listdir(input_dir):
    # Load the image
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    # Apply the augmentation to the image
    augmented_image = augmentation(image=img)

    # Save the augmented image
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, augmented_image)

    print(f"Augmented {filename} and saved to {output_path}")