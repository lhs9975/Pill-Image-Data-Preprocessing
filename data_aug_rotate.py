# 이미지 회전

import imgaug.augmenters as iaa
import os
import cv2

input_dir = 'D:\\pill\\image\\EXK\\ee'
output_dir = 'D:\\pill\\image\\EXK\\rotate'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# define rotation augmenter
rotate_aug = iaa.Affine(rotate=(-15, 15))

# iterate through images in input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        # apply rotation augmentation and save rotated images
        for i in range(30):
            rotated_img = rotate_aug.augment_image(img)
            output_path = os.path.join(output_dir, f'{filename[:-4]}_{i}.jpg')
            cv2.imwrite(output_path, rotated_img)

print(f"Processed {filename} and saved to {output_path}")