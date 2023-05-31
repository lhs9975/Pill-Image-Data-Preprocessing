# Black Hat / 전처리 최종본

import cv2
import os

# Gray Scale, Histogram Equization, desoise 동시 수행

# 다비드정100mg
# Directory containing the input images
input_dir = "E:\\rembg_data\\david"

# Directory where the output images will be saved
output_dir = "E:\\blackhat_data\\david_blackhat"

# 값 범위 조절 필요, 커널의 모형으로 MORPH_RECT = 직사각형
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))

# Iterate over all the files in the input directory
for filename in os.listdir(input_dir):
    # image load
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    # Gray Scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # histogram equalization
    equ = cv2.equalizeHist(gray)

    # Denoise
    # h : 필터의 강도를 결정하는 파라미터, 클수록 노이즈를 더 잘 제거, 단 너무 크면 이미지의 디테일도 제거
    # h값 조절 필요
    denoised = cv2.fastNlMeansDenoising(equ, None, h=20, templateWindowSize=7, searchWindowSize=21)

    # Perform Black Hat operation
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)

    # Save the output image
    out_path = os.path.join(output_dir, 'bh_' + filename)
    cv2.imwrite(out_path, blackhat)

    print(f"Processed {filename} and saved to {out_path}")