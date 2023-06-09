import cv2 as cv
import numpy as np

def GetYValueListInContour(contour, xPos):
    '''
    xPos에 해당하는 y좌표를 contour에서 찾아 반환한다.
    '''
    yList = []
    for each in contour:
        currentXPos = each[0][0]
        currentYPos = each[0][1]
        if currentXPos == xPos:
            yList.append(currentYPos)

    return yList

def GetMinMaxPosInContour(contour, maskedImg):
    global minX,minY,maxX,maxY
    '''
    contour의 x,y 좌표에서 최대 최소, 중심 좌표를 반환한다.
    :param contour:
    :param cx:
    :param cy:
    :return:
    '''

    # 중심점을 구한다.
    M = cv.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    '''
    cv.circle(maskedImg, (cx, cy), 10, (0, 255, 255), -1)
    cv.imshow("maskedImg", maskedImg)
    cv.waitKey(0)
    '''
    # 초기값으로 외곽선의 중심 좌표를 넣어 준다.
    minX,minY,maxX,maxY = cx, cy, cx, cy

    for each in contour:
        currentX = each[0][0]
        currentY = each[0][1]
        if currentX < minX:
            minX = currentX
        elif currentX > maxX:
            maxX = currentX

        if currentY < minY:
            minY = currentY
        elif currentY > maxY:
            maxY = currentY

    return minX,minY,maxX,maxY, cx, cy

def GetPillContour(img_color, highThreshold = 50, showFlag = False):
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    img_gray = cv.medianBlur(img_gray, 5)
    if showFlag:
        cv.imshow("Blur", img_gray)
        cv.waitKey(0)

    img_colorTmp = img_color.copy()

    low = 0
    img_canny = cv.Canny(img_gray, low, highThreshold)
    if showFlag:
        cv.imshow("img_canny", img_canny)
        cv.waitKey(0)

    adaptive = cv.adaptiveThreshold(img_canny, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 0)

    # contour에서 구멍 난 경우를 대비
    kernel = np.ones((10, 10), np.uint8)
    closing = cv.morphologyEx(adaptive, cv.MORPH_CLOSE, kernel)
    contourInput = closing
    contours, hierarchy = cv.findContours(contourInput, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    maxIdx = 0
    idx = 0
    maxArea = 0.0

    for cnt in contours:
        area = cv.contourArea(cnt)
        if maxArea < area:
            maxIdx = idx
            maxArea = area
        idx = idx + 1

    cv.drawContours(img_colorTmp, contours, idx - 1, (0, 0, 255), 1)
    if showFlag:
        cv.imshow("img_colorTmp", img_colorTmp)
        cv.waitKey(0)

        # 알약 외곽선을 추출한다.
    #contour = contours[idx - 1]
    lastIdx = idx - 1
    return contours, lastIdx

def hangulFilePathImageRead ( filePath ) :

    stream = open( filePath.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)

    return cv.imdecode(numpyArray , cv.IMREAD_UNCHANGED)

def GetForegroundImgInfo(FilePath, showFlag = False):
    '''
    전경이미지와 전경이미지를 얻을떄 사용한 정보들을 반환
    :param FilePath:
    :param showFlag:
    :return:
    '''
    #import time
    #start = time.time()
    #print(FilePath)
    img_color = cv.imread(FilePath)
    #cv.imshow('ss', img_color)
    #cv.waitKey(0)
    #img_color = hangulFilePathImageRead(FilePath)
    img_color = cv.resize(img_color, (300,300))
    #cv.imshow('ss', img_color)
    #cv.waitKey(0)
    # 알약 외곽선을 추출한다.
    highThreshold = 100
    contours, lastIdx = GetPillContour(img_color =img_color, highThreshold = highThreshold, showFlag = showFlag)
    '''
    contourLen = len(contours)
    if showFlag:
        print("contour len : ")
        print(contourLen)

    loopCount = 0
    while contourLen > 1 :
        highThreshold -= 150
        contours, lastIdx = GetPillContour(img_color =img_color, highThreshold = highThreshold, showFlag = showFlag)
        contourLen = len(contours)
        if showFlag:
            print("contour len : ")
            print(contourLen)

        loopCount += 1
        if loopCount > 3 :
            raise Exception('contour Length is not 1')

    '''

    if contours is None:
        print("contour is not found!!")
    # 검정색 배경을 만듦
    ImgMasking = np.zeros((img_color.shape[0], img_color.shape[1], 3), np.uint8)

    minX,minY,maxX,maxY, cx, cy = GetMinMaxPosInContour(contours[lastIdx], ImgMasking)

    '''
    cv.circle(ImgMasking, (cx, cy), 10, (0, 255, 255), -1)
    cv.imshow("center point", ImgMasking)
    cv.waitKey(0)
    '''
    # 원본 이미지에서 찾은 컨투어의 좌표를 위에서 만든 검정 배경에 그림
    cv.drawContours(ImgMasking, contours, lastIdx, (255, 255, 255), 2)
    if showFlag:
        cv.imshow("ImgMasking1", ImgMasking)
        cv.waitKey(0)

    # 내부를 흰색으로 채워 줌
    mask = np.zeros((img_color.shape[0] + 2, img_color.shape[1] + 2), np.uint8)
    mask[:] = 0
    cv.floodFill(ImgMasking, mask, (cx, cy), (255, 255, 255))
    #print(cx,cy)

    dst = cv.copyTo(img_color, ImgMasking)

    #print("time :", time.time() - start)
    if showFlag:
        cv.imshow("ImgMasking2", ImgMasking)
        cv.imshow("Last", dst)
        cv.waitKey(0)

    return dst, ImgMasking, contours, lastIdx, cx, cy, minX,minY,maxX,maxY

def IsItOutside(x1, y1, x2, y2):
    '''
    좌표가 알약 외부에 있는지 확인한다.
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return: 알약 외부에 있으면 True
    '''
    retValue = False
    if (minX > x1) or (maxX < x1):
        retValue = True
    elif (minX > x2) or (maxX < x2):
        retValue = True
    elif (minY > y1) or (maxY < y1):
        retValue = True
    elif (minY > y2) or (maxY < y2):
        retValue = True

    return  retValue


def CalcPillPositionInfo(FilePath, showFlag = False):
    global foreGroundImg, ImgMasking, contours, lastIdx, cx, cy, minX, minY, maxX, maxY, splitLineInfo, minXOfSplitLine, maxXOfSplitLine, minYOfSplitLine, maxYOfSplitLine

    foreGroundImg, ImgMasking, contours, lastIdx, cx, cy, minX, minY, maxX, maxY = GetForegroundImgInfo(
        FilePath, showFlag)


if __name__ == '__main__':
    import glob
    import os.path as path

    DataParentPath = 'E:\\data\\train\\29002\\'
    fileName = 'IMG_20201202_163857.png'
    FilePath = DataParentPath + fileName
    FlagOfExcuteKind = True
    if FlagOfExcuteKind:
        dst, ImgMasking, contours, lastIdx, cx, cy, minX, minY, maxX, maxY = GetForegroundImgInfo(
            FilePath, False)
        cv.waitKey(0)

    else:

        TargetPath = DataParentPath
        FilePathList = glob.glob(path.join(TargetPath, '*.*'))

        for each in FilePathList:
            print(each)

            dst, ImgMasking, contours, lastIdx, cx, cy, minX, minY, maxX, maxY = GetForegroundImgInfo(
                each)

#savePath = DataParentPath + "\\mask\\" + fileName
   #cv.imwrite(savePath, dst)
