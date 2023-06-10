# # 배경 제거

# from rembg import remove 

# # PIL 패키지에서 Image 클래스 불러오기
# from PIL import Image 

# # 입력 이미지 불러오기
# for i in range(1,6,1):
#     img = Image.open("C:\\Users\\LEE\\Desktop\\original_data\\gas\\front\\IMG-%04d.png" % i)


# # 배경 제거하기
#     out = remove(img)


# # 변경된 이미지 저장하기
#     out.save("E:\\rembg_data\\gas\\front\\gas-%04d.png" % i)

import os
import rembg

def remove_background_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            jpg_file_path = os.path.join(folder_path, filename)
            output_file_path = os.path.join(folder_path, "rembg_" + filename)
            remove_background(jpg_file_path, output_file_path)

def remove_background(input_file_path, output_file_path):
    with open(input_file_path, "rb") as input_file, open(output_file_path, "wb") as output_file:
        input_image = input_file.read()
        output_image = rembg.remove(input_image)
        output_file.write(output_image)

# Example usage
folder_path = "E:\\Pill Project\\resize_data\\pilcon"

remove_background_folder(folder_path)