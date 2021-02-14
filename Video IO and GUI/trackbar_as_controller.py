import cv2

maxScaleUp = 100
scaleValue = 1
scaleType = 0
maxType = 1
scaleFactor = 1.0

windowName = "Resize Image"
trackbarValue = "Scale"
trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down"

# load an image
im = cv2.imread("truth.png")

# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

def displayImage():
    global scaleType
    global scaleValue
    global scaleFactor
    if scaleType == 1:
        scaleFactor = 1 - scaleValue/100.0
    else:
        scaleFactor = 1 + scaleValue/100.0
    if scaleFactor == 0: # clip the scale factor
        if scaleType == 1:
            scaleFactor = 0.01
        else:
            scaleFactor = 1
    scaledImage = cv2.resize(im, None, fx=scaleFactor,\
            fy = scaleFactor, interpolation = cv2.INTER_LINEAR)
    cv2.imshow(windowName, scaledImage)

# Callback functions
def scaleTypeImage(*args):
    global scaleType
    scaleType = args[0]
    displayImage()

def scaleImage(*args):
    global scaleValue
    scaleValue = args[0]
    displayImage()

cv2.createTrackbar(trackbarValue, windowName, scaleValue, maxScaleUp, scaleImage)
cv2.createTrackbar(trackbarType, windowName, scaleType, maxType, scaleTypeImage)
scaleImage(1)
while True:
    c = cv2.waitKey(10)
    if c==27:
        break

cv2.destroyAllWindows()
