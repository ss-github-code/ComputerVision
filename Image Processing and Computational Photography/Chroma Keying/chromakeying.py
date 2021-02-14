# -*- coding: utf-8 -*-
"""
Submission for chromakey
"""
def onMouse(action, x, y, flags, userdata):
    global frameHsv, panel, keyColorHsvList, keyColor, keyColorHsv

    if action == cv2.EVENT_LBUTTONUP:
        # append the currently selected color to the list
        keyColorHsvList.append(frameHsv[y, x])
        #print('current:', frame[y, x], frameHsv[y, x])
        
        # update the keyColor as the mean of the selected colors
        arr = np.array(keyColorHsvList)
        color = arr.mean(axis=0)
        
        keyColorHsv = tuple(np.uint8(c) for c in color)
        keyColor = cv2.cvtColor(np.array([[keyColorHsv]]), 
                                   cv2.COLOR_HSV2BGR)
        
        keyColor = tuple(int(c) for c in keyColor[0,0])
        #print('mean:', keyColor, keyColorHsv)
        # update the color patch selector window
        cv2.rectangle(panel, (0, 0), (100, 100), keyColor, -1)
        cv2.imshow(panelName, panel)
        
def apply(x):
    global show
    applyFilter(show)

def clipHsv(hsv):
    h = np.clip(hsv[0], 0, 180)
    s = np.clip(hsv[1], 0, 255)
    v = np.clip(hsv[2], 0, 255)
    return np.array([h, s, v])

def applyFilter(showBkImg):
    global tolTrackBarHue, tolTrackBarSat, tolTrackBarVal, blurTrackBar
    global windowName, keyColorHsv, kernel
    # find the tolerances and the kernel size for the blur filter
    tolH = cv2.getTrackbarPos(tolTrackBarHue, windowName)
    tolS = cv2.getTrackbarPos(tolTrackBarSat, windowName)
    tolV = cv2.getTrackbarPos(tolTrackBarVal, windowName)
    blur = cv2.getTrackbarPos(blurTrackBar, windowName)
    
    # calculate the low and high threshold
    low = np.array([keyColorHsv[0] - tolH, keyColorHsv[1] - tolS, keyColorHsv[2] - tolV])
    high = np.array([keyColorHsv[0] + tolH, keyColorHsv[1] + tolS, keyColorHsv[2] + tolV])
    
    low = clipHsv(low)
    high = clipHsv(high)

    #print('low', low)
    #print('high', high)        
    
    mask = cv2.inRange(frameHsv, low, high) # mask based on threshold
    fg_mask = cv2.bitwise_not(mask)
    #bg_mask = cv2.bitwise_not(fg_mask)
    
    blurKernelSize = 2*blur+1 # blur the mask
    fg_mask = cv2.GaussianBlur(fg_mask, (blurKernelSize, blurKernelSize), 0)
    
    # dilate the fg_mask
    #fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)
    
    fg = cv2.bitwise_and(frame, frame, mask = fg_mask)
    bg_mask = cv2.bitwise_not(fg_mask) # background mask
    
    bg = frame - fg
    if showBkImg == 1:
        #print('merge')
        f = cv2.copyTo(bg_image, bg_mask, fg)
        cv2.imshow(windowName, f)
    elif showBkImg == 2: # fg image
        #print('fg')
        cv2.imshow(windowName, fg)
    else:
        #print('not fg')
        cv2.imshow(windowName, bg)

if __name__ == '__main__':
    import cv2
    import numpy as np

    # open the video with the green screen background
    cap = cv2.VideoCapture('greenscreen-demo.mp4')
    ret, frame = cap.read() 
    # read the background image (same res as the video)
    bgImage = cv2.imread('bg_1920x1080.jpg')
    
    # resize the frame to fit the screen along with the trackbar
    frame = cv2.resize(frame, (1280, 720))
    frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # HSV
    # resize the background image
    bg_image = cv2.resize(bgImage, (1280, 720))
    
    # setup the GUI
    windowName = 'Screen'
    panelName = 'Color Patch'
    tolTrackBarHue = 'Tolerance Hue'
    tolTrackBarSat = 'Tolerance Sat'
    tolTrackBarVal = 'Tolerance Val'
    blurTrackBar = 'Blur'
    
    panel = np.zeros([100, 100, 3], np.uint8) # color patch window
    keyColorHsvList = []
    
    keyColor = (60, 175, 75)  # initial value for key color and tolerance
    keyColorHsv = (56, 168, 175)
    
    cv2.namedWindow(windowName)
    cv2.namedWindow(panelName)
    cv2.createTrackbar(tolTrackBarHue, windowName, 30, 180, apply)
    cv2.createTrackbar(tolTrackBarSat, windowName, 80, 255, apply)
    cv2.createTrackbar(tolTrackBarVal, windowName, 80, 255, apply)
    cv2.createTrackbar(blurTrackBar, windowName, 3, 10, apply)
    cv2.imshow(panelName, panel)

    cv2.setMouseCallback(windowName, onMouse)

    kernelSize = 5        
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernelSize, kernelSize))
    
    cv2.imshow(windowName, frame)    
    k = 0
    wait = 0
    show = 0 # 0: fg image only, 1: bk image, 2: mask

    while ret:
        k = cv2.waitKey(wait)
        if k == 27: # esc
            break
        elif k == 99:  # c: continue
            wait = 40
            show = 1
            applyFilter(show)
            continue
        # apply, show mask, next frame, pause
        elif k == 97 or k == 109 or k== 110 or k == 112 or k == 114: # a: apply, m: mask, p: pause
            wait = 0
            if k == 112:
                print("bg")
                show = 0
            elif k == 109:
                print("fg")
                show = 2
            elif k == 114:
                print("set bg")
                show = 1
            elif k == 110:
                print("next")
            else:
                print("apply")
            applyFilter(show)
            if k != 110:
                continue
        else:
            applyFilter(show)

        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720))
            frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    cap.release()
    cv2.destroyAllWindows()
