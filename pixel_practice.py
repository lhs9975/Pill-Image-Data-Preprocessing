# 픽셀 변환은 스마트폰으로 촬영 시 크게 축소되므로 잠시 보류

# import cv2

# # Load the image
# img = cv2.imread('D:\\pill\\image\\EXK\\ee\\EXK1-00001.jpg')

# # Resize the image to 75x75 pixels
# resized_img = cv2.resize(img, (75, 75))

# # Save the resized image
# cv2.imwrite('D:\\pill\\image\\EXK\\pixel\\EXK1-00001.jpg', resized_img)

# import cv2

# # Load the input image
# img = cv2.imread('EXK1-00001.jpg')

# # Check that the input image is valid
# if img is None or img.shape[0] == 0 or img.shape[1] == 0:
#     print('Error: invalid input image')
# else:
#     # Resize the image to 75x75 pixels
#     img_resized = cv2.resize(img, (75, 75))

#     # Save the resized image
#     cv2.imwrite('EXK1-00001.jpg', img_resized)

# 아래 코드 사용 시 전체적인 이미지 축소로 인해 화질 저하
import os
import cv2

input_dir = "D:\\pill\\image\\EXK\\ee"
output_dir = "D:\\pill\\image\\EXK\\pixel"

# Loop over all files in the input directory
for filename in os.listdir(input_dir):
    # Read the image
    image = cv2.imread(os.path.join(input_dir, filename))
    # Resize the image to 75x75 pixels
    resized_image = cv2.resize(image, (3000, 3000))
    # Save the resized image to the output directory
    cv2.imwrite(os.path.join(output_dir, '1_' + filename), resized_image)


# 아래 코드 사용 시 전체적인 픽셀 감소 화질 저하 없음
import os
import cv2
import imgaug.augmenters as iaa

input_dir = "D:\\pill\\image\\EXK\\ee"
output_dir = "D:\\pill\\image\\EXK\\size"

# Define augmenter that resizes images to a random scale between 50% and 100% of their original size
resize_augmenter = iaa.Resize((0.02, 0.02))

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