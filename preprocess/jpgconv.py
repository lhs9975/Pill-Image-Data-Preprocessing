import os
import glob

files = glob.glob("C:\\Users\\LEE\\Desktop\\original_data\\알레그라정120mg\\*.jpg")
for name in files:
    if not os.path.isdir(name):
        src = os.path.splitext(name)
        os.rename(name,src[0]+'.png')