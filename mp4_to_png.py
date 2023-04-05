import os
import cv2

vidcap = cv2.VideoCapture('EXSM.MP4') # (영상 파일 이름 입력)
success, image = vidcap.read()

count = 1

while success:
    cv2.imwrite("image_2\EXSM\EXSM-%05d.png" % count, image) # (-%05d 앞에 경로 지정)
    success, image = vidcap.read()
    print('Read a new frame : ', count)
    count += 1
    
print("비디오 내 이미지 png 파일 변환 완료")