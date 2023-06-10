import cv2
import os

# Simultaneous execution of Gray Scale, Histogram Equalization, and Denoise

# David Tab. 100mg
# Directory containing the input images
input_dir = "E:\\Pill Project\\rembg_data\\ace\\b"

# Directory where the output images will be saved
output_dir = "E:\\Pill Project\preprocess_data\\a"

# Need to adjust value range, model of kernel as MORPH_RECT = rectangle
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))

# Iterate over all the files in the input directory
for filename in os.listdir(input_dir):
    # Load image
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    # Gray Scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Denoise (using Gaussian filter)
    # (5, 5) 매개변수는 필터의 커널 크기를 지정하고 0은 커널 크기를 기준으로 가우시안 커널의 표준 편차가 자동으로 계산됨을 나타냄
    denoised = cv2.GaussianBlur(clahe_img, (1, 1), 0)

    # Perform Black Hat operation
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)

    # Save the output image
    out_path = os.path.join(output_dir, 'bhb_' + filename)
    cv2.imwrite(out_path, blackhat)

    print(f"Processed {filename} and saved to {out_path}")