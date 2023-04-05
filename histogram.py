import cv2
import os

# Gray Scale 과 Histogram Equization 동시 수행

# Directory containing the input images
input_dir = "D:\\pill\\image\\EXK\\EXK2(rembg)"

# Directory where the output images will be saved
output_dir = "D:\\pill\\image\\EXK\\EXK8(rembg_his)"

# Iterate over all the files in the input directory
for filename in os.listdir(input_dir):
    # Load the image
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equ = cv2.equalizeHist(gray)
    
    # Save the output image
    out_path = os.path.join(output_dir, 'origianl_' + filename)
    cv2.imwrite(out_path, equ)

    print(f"Processed {filename} and saved to {out_path}")
