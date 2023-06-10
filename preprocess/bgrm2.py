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
folder_path = "E:\\Pill Project\\data_test3\\rexo"

remove_background_folder(folder_path)
