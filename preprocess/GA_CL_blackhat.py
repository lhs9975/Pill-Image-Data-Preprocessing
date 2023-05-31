import cv2
import os

# Simultaneous execution of Gray Scale, Histogram Equalization, and Denoise

# David Tab. 100mg
# Directory containing the input images
input_dir = "E:\\rembg_data\\david"

# Directory where the output images will be saved
output_dir = "E:\\blackhat_data\\david_clahe"

# Need to adjust value range, model of kernel as MORPH_RECT = rectangle
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))

# Iterate over all the files in the input directory
for filename in os.listdir(input_dir):
    # Load image
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    # Gray Scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Denoise (using Gaussian filter)
    denoised = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # Perform Black Hat operation
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)

    # Save the output image
    out_path = os.path.join(output_dir, 'bh_' + filename)
    cv2.imwrite(out_path, blackhat)

    print(f"Processed {filename} and saved to {out_path}")
