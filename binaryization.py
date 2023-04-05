# 이진화 작업

import os
import cv2

# Set the directory where the images are stored
# dir_path_i = 입력 이미지 폴더 경로
# dir_path_o = 출력 이미지 폴더 경로
dir_path_i = 'D:\\pill\\image\\EXK\\EXK2(rembg)'
dir_path_o = 'D:\\pill\\image\\EXK\\EXK4(rembg_bin)'

# Loop through all files in the directory
for filename in os.listdir(dir_path_i):

    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image in grayscale
        img = cv2.imread(os.path.join(dir_path_i, filename), cv2.IMREAD_GRAYSCALE)

        # Binarize the image using thresholding
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Save the binarized image
        cv2.imwrite(os.path.join(dir_path_o, 'rembg_' + filename), binary_img)

    print(f"Processed {filename} and saved to {dir_path_o}")