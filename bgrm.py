# 배경 제거

from rembg import remove 

# PIL 패키지에서 Image 클래스 불러오기
from PIL import Image 

# 입력 이미지 불러오기
for i in range(1,7,1):
    img = Image.open("D:\\pill\\image\\three\\original\\EX-%04d.png" % i)


# 배경 제거하기
    out = remove(img) 


# 변경된 이미지 저장하기
    out.save("D:\\pill\\image\\three\\bgrm\\EX-%04d.png" % i)