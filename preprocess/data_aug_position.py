# 위치 조절

import imgaug.augmenters as iaa
import os
import cv2

input_dir = "E:\\Pill Project\\resize_data\\pilcon"
output_dir = "E:\\Pill Project\\preprocess_data\\b"

# Create augmenter to adjust the vertical position by 1%
# x, y 값 -> x 우측 : 양수, 좌측 : 음수 ||||    y 상 : 음수, 하 : 양수
augmenter1 = iaa.Affine(translate_percent={"x": 0.1, "y": -0.1})

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    # Load image
    img = cv2.imread(os.path.join(input_dir, filename))

    # Apply augmentation
    img_aug = augmenter1.augment_image(img)

    # Save augmented image to output directory
    cv2.imwrite(os.path.join(output_dir, 'po1_' + filename), img_aug)


augmenter2 = iaa.Affine(translate_percent={"x": -0.1, "y": -0.1})

for filename in os.listdir(input_dir):
    # Load image
    img = cv2.imread(os.path.join(input_dir, filename))

    # Apply augmentation
    img_aug = augmenter2.augment_image(img)

    # Save augmented image to output directory
    cv2.imwrite(os.path.join(output_dir, 'po2_' + filename), img_aug)


augmenter3 = iaa.Affine(translate_percent={"x": 0.1, "y": 0.1})

for filename in os.listdir(input_dir):
    # Load image
    img = cv2.imread(os.path.join(input_dir, filename))

    # Apply augmentation
    img_aug = augmenter3.augment_image(img)

    # Save augmented image to output directory
    cv2.imwrite(os.path.join(output_dir, 'po3_' + filename), img_aug)


augmenter4 = iaa.Affine(translate_percent={"x": -0.1, "y": 0.1})

for filename in os.listdir(input_dir):
    # Load image
    img = cv2.imread(os.path.join(input_dir, filename))

    # Apply augmentation
    img_aug = augmenter4.augment_image(img)

    # Save augmented image to output directory

    cv2.imwrite(os.path.join(output_dir, 'po4_' + filename), img_aug)