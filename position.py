# 위치 조절

import imgaug.augmenters as iaa
import os
import cv2

input_dir = "D:\\pill\\image\\EXK\\ee"
output_dir = "D:\\pill\\image\\EXK\\position"

# Create augmenter to adjust the vertical position by 1%
# x, y 값 -> x 우측 : 양수, 좌측 : 음수  y 상 : 음수, 하 : 양수
augmenter = iaa.Affine(translate_percent={"x": 0.1})

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    # Load image
    img = cv2.imread(os.path.join(input_dir, filename))

    # Apply augmentation
    img_aug = augmenter.augment_image(img)

    # Save augmented image to output directory
    cv2.imwrite(os.path.join(output_dir, filename), img_aug)
