import cv2
import os

def calculate_average_brightness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = cv2.mean(gray_image)[0]
    return average_brightness

def calculate_average_brightness_folder(folder_path):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            average_brightness = calculate_average_brightness(image)
            print(f"Image: {filename}, Average Brightness: {average_brightness}")

# Provide the path to the folder containing the images
folder_path = "C:\\Users\\LEE\\Desktop\\alle"
calculate_average_brightness_folder(folder_path)
