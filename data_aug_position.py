# 위치 조절

import imgaug.augmenters as iaa
import os
import cv2

input_dir = "D:\\datasets\\training_set\\EX2"
output_dir = "C:\\Users\\LHS\\Desktop\\pixel"

# Create augmenter to adjust the vertical position by 1%
# x, y 값 -> x 우측 : 양수, 좌측 : 음수 ||||    y 상 : 음수, 하 : 양수
augmenter = iaa.Affine(translate_percent={"x": -0.2, "y": 0.2})

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    # Load image
    img = cv2.imread(os.path.join(input_dir, filename))

    # Apply augmentation
    img_aug = augmenter.augment_image(img)

    # Save augmented image to output directory
    cv2.imwrite(os.path.join(output_dir, 'position-4' + filename), img_aug)
