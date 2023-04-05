# imagug 라이브러리의 rotate 연습 코드

# import os
# import cv2
# import imgaug.augmenters as iaa

# # set the directory path where the image is stored
# input_dir = "D:\\pill\\image\\EXK\\ee"
# output_dir = "D:\\pill\\image\\EXK\\rotate"

# # define the augmenter with multiple angles
# aug = iaa.Affine(rotate=(-10, 10))

# # loop to create 20 augmented images with different angles
# for i in range(20):
#     # read the original image
#     image = cv2.imread(os.path.join(input_dir))

#     # apply the augmentation
#     aug_image = aug(image=image)

#     # save the augmented image
#     cv2.imwrite(os.path.join(output_dir, 'aug_' + str(i) + '_'), aug_image)

# import os
# import cv2
# import imgaug.augmenters as iaa

# input_dir = 'D:\\pill\\image\\EXK\\ee'
# output_dir = 'D:\\pill\\image\\EXK\\rotate'

# # create augmenter to rotate images from -10 to 10 degrees
# augmenter = iaa.Affine(rotate=(-10, 10))

# # iterate over all files in the input directory
# for filename in os.listdir(input_dir):
#     # load the image
#     image_path = os.path.join(input_dir, filename)
#     image = cv2.imread(image_path)

#     # apply rotation augmentation to image
#     rotated_images = [augmenter.augment_image(image) for _ in range(20)]

#     # save rotated images to the output directory
#     for i, rotated_image in enumerate(rotated_images):
#         output_path = os.path.join(output_dir, f"{filename.split('.')[0]}_rotated_{i}.jpg")
#         cv2.imwrite(output_path, rotated_image)

import imgaug.augmenters as iaa
import cv2
import os

# Set the directories
input_dir = 'D:\\pill\\image\\EXK\\ee'
output_dir = 'D:\\pill\\image\\EXK\\rotate'

# Create the augmenter
seq = iaa.Sequential([
    iaa.Rotate((-10, 10))
])

# Loop through each file in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
        
        # Load the image
        image = cv2.imread(os.path.join(input_dir, file_name))

        # Apply the augmenter 20 times and save each image
        for i in range(20):
            augmented_image = seq.augment_image(image)
            new_file_name = file_name.split(".")[0] + "_augmented_" + str(i) + ".jpg"
            cv2.imwrite(os.path.join(output_dir, new_file_name), augmented_image)


import os
import cv2
import imgaug.augmenters as iaa

# Set input and output directories
input_dir = 'D:\\pill\\image\\EXK\\ee'
output_dir = 'D:\\pill\\image\\EXK\\rotate'

# Get a list of all image files in the input directory
img_files = os.listdir(input_dir)
img_files = [f for f in img_files if f.endswith('.jpg') or f.endswith('.png')]

# Define the augmentation sequence
seq = iaa.Sequential([
    iaa.Affine(rotate=range(-10, 11))
])

# Apply the augmentation and save the resulting images
for f in img_files:
    # Load the image
    img = cv2.imread(os.path.join(input_dir, f))
    # Apply the augmentation
    images_aug = seq(images=[img]*20)
    # Save the resulting images
    for i, img_aug in enumerate(images_aug):
        filename, ext = os.path.splitext(f)
        output_file = f"{filename}_{i}{ext}"
        cv2.imwrite(os.path.join(output_dir, output_file), img_aug)

    print(f"Processed {filename} and saved to {output_file}")

import os
import cv2
import imgaug.augmenters as iaa

# set input and output directories
input_dir = 'D:\\pill\\image\\EXK\\ee'
output_dir = 'D:\\pill\\image\\EXK\\rotate'

# set the range of angles to rotate the image
rotate_range = range(-10, 11)

# define the augmenter with sequential rotation by 1 degree
seq = iaa.Sequential([iaa.Rotate(angle=i) for i in rotate_range])

# loop through images in input directory
for filename in os.listdir(input_dir):
    # read image
    img = cv2.imread(os.path.join(input_dir, filename))
    
    # apply augmentation
    images_aug = seq(images=[img])
    
    # save augmented images
    for idx, image_aug in enumerate(images_aug):
        output_path = os.path.join(output_dir, f"{filename.split('.')[0]}_{rotate_range[idx]}.jpg")
        cv2.imwrite(output_path, image_aug)

import imgaug.augmenters as iaa
import os
import cv2

input_dir = 'D:\\pill\\image\\EXK\\ee'
output_dir = 'D:\\pill\\image\\EXK\\rotate'

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
            output_path = os.path.join(output_dir, f'{filename[:-4]}_{i}.jpg')
            cv2.imwrite(output_path, rotated_img)

print(f"Processed {filename} and saved to {output_path}")