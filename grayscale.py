import os
import cv2

# Set the directory where the images are stored
# dir_path_i = 입력 이미지 폴더 경로
# dir_path_o = 출력 이미지 폴더 경로
dir_path_i = 'D:\\pill\\image\\EXK\\EXK2(rembg)'
dir_path_o = 'D:\\pill\\image\\EXK\\EXK6'

# Loop through all files in the directory
for filename in os.listdir(dir_path_i):

    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image in grayscale
        img = cv2.imread(os.path.join(dir_path_i, filename), cv2.IMREAD_GRAYSCALE)

        # Save the binarized image
        cv2.imwrite(os.path.join(dir_path_o, 'rembg_' + filename), img)

print(f"Processed {filename} and saved to {dir_path_o}")