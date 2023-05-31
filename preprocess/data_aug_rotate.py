# 이미지 회전

import imgaug.augmenters as iaa
import os
import cv2

input_dir = 'E:\\data\\valid\\imprint\\datasets\\TRA'
output_dir = 'C:\\Users\\LEE\\Desktop\\new'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# define rotation augmenter
rotate_aug = iaa.Affine(rotate=(-10, 10))

# iterate through images in input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        # apply rotation augmentation and save rotated images
        for i in range(20):
            rotated_img = rotate_aug.augment_image(img)
            output_path = os.path.join(output_dir, f'{"rotate"[:5]}_{i}.jpg')
            cv2.imwrite(output_path, rotated_img)

print(f"Processed {filename} and saved to {output_path}")