# 논문에서 말하는 크기 조절이 픽셀 조절인 것 같음
# 이부분은 좀 더 고려

# import os
# import cv2
# import imgaug.augmenters as iaa

# input_dir = "D:\\pill\\image\\EXK\\ee"
# output_dir = "D:\\pill\\image\\EXK\\size"

# # Define augmenter
# aug = iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)})

# # Loop through images in input directory
# for filename in os.listdir(input_dir):
#     # Read image
#     img = cv2.imread(os.path.join(input_dir, filename))
#     # Apply augmentation
#     img_aug = aug(image=img)
#     # Save image
#     cv2.imwrite(os.path.join(output_dir, filename), img_aug)



# import imgaug.augmenters as iaa
# import cv2
# import os

# input_dir = "D:\\pill\\image\\EXK\\ee"
# output_dir = "D:\\pill\\image\\EXK\\size"

# # Create a directory to store output images
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# images = os.listdir(input_dir)

# # Define augmenter
# seq = iaa.Sequential([iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)})])

# for image in images:
#     img_path = os.path.join(input_dir, image)
#     img = cv2.imread(img_path)

#     # Apply augmenter with a difference of 1% scaling between -10% and 10%
#     for i in range(-10, 11):
#         scale = i / 100 + 1
#         augmented_img = seq.augment_image(img, scale=scale)
#         output_path = os.path.join(output_dir, f"{image.split('.')[0]}_{i+10}.jpg")
#         cv2.imwrite(output_path, augmented_img)



# import imgaug.augmenters as iaa
# import os
# import cv2

# input_dir = "D:\\pill\\image\\EXK\\ee"
# output_dir = "D:\\pill\\image\\EXK\\size"

# # create output_dir if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # define augmenter
# aug = iaa.Affine(scale={"x": (1.5, 0.2), "y": (0.2, 1.5)})

# # loop through images in input_dir
# for filename in os.listdir(input_dir):
#     # read image
#     img = cv2.imread(os.path.join(input_dir, filename))
#     # apply augmentation
#     img_aug = aug(image=img)
#     # save augmented image
#     for i in range(-10, 11):
#         scale = 1 + i / 100.0
#         filename_aug = os.path.splitext(filename)[0] + "_scale_{:.2f}.jpg".format(scale)
#         cv2.imwrite(os.path.join(output_dir, filename_aug), img_aug)



# import imgaug.augmenters as iaa
# import os
# import cv2

# input_dir = 'D:\\pill\\image\\EXK\\bb'
# output_dir = 'D:\\pill\\image\\EXK\\size'

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # define rotation augmenter
# scale_aug = iaa.Resize((0.5, 1.0))

# # iterate through images in input directory
# for filename in os.listdir(input_dir):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         img_path = os.path.join(input_dir, filename)
#         img = cv2.imread(img_path)

#         # apply rotation augmentation and save rotated images
#         for i in range(20):
#             scale_img = scale_aug.augment_image(img)
#             output_path = os.path.join(output_dir, f'{filename[:-4]}_{i}.jpg')
#             cv2.imwrite(output_path, scale_img)



import os
import cv2
import imgaug.augmenters as iaa

input_dir = "D:\\pill\\image\\EXK\\ee"
output_dir = "D:\\pill\\image\\EXK\\size"

# Define augmenter that resizes images to a random scale between 50% and 100% of their original size
resize_augmenter = iaa.Resize((0.5, 0.5))

# Loop through each image in the input directory
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    
    # Load the image
    image = cv2.imread(input_path)
    
    # Apply the resize augmenter to the image
    resized_image = resize_augmenter.augment_image(image)
    
    # Save the resized image to the output directory
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, resized_image)
