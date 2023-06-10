import os
import glob

files = glob.glob("E:\\Pill Project\\resize_data\\hue\\f\\*.jpg")
for name in files:
    if not os.path.isdir(name):
        src = os.path.splitext(name)
        os.rename(name,src[0]+'.png')