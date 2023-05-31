import os
from PIL import Image

# 이미지가 저장된 폴더 경로
image_folder = 'E:\\datasets\\test_set\\EX1'

# 리사이즈할 크기
size = (75, 75)

# 폴더 내 모든 이미지 파일에 대해 반복문 실행
for filename in os.listdir(image_folder):
    # 이미지 파일 열기
    image = Image.open(os.path.join(image_folder, filename))
    
    # 이미지 리사이즈하기
    resized_image = image.resize(size)
    
    # 리사이즈된 이미지 저장하기
    resized_image.save(os.path.join(image_folder, 'resized_' + filename))