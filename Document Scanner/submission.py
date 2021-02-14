# Submission for document scanner
import numpy as np
import cv2

# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

img = cv2.imread('grabcut_output.png')

# convert to gray scaler, blur it, and find edges in the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edge = cv2.Canny(gray, 75, 200)

#cv2.imshow("Image", img)
#cv2.imshow("Edge", edge)

# find the contours in the edged image, keeping only the largest one
cnts, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = lambda x:cv2.contourArea(x), reverse = True)[:5]

for c in cnts:
    # contour approximation with small epsilon (2% of arc length)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 4: # found the rectangle
        mainEdge = approx
        break

#cv2.drawContours(img, [mainEdge], -1, (0, 255, 0), 2)
#cv2.imshow("Image", img)

def order_points(pts):
    pts = pts.reshape((4,2))
    # order the points top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype=np.float32)
    add = pts.sum(axis=1) # x + y
    rect[0] = pts[np.argmin(add)] # top-left has smallest sum
    rect[2] = pts[np.argmax(add)] # bottom-right has largest sum
    diff = np.diff(pts, axis=1) # y - x
    rect[1] = pts[np.argmin(diff)] # top-right has smallest diff
    rect[3] = pts[np.argmax(diff)] # bottom-left has largest diff
    return rect

def four_point_transform(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(np.power(br[0] - bl[0], 2) + np.power(br[1] - bl[1], 2))
    widthB = np.sqrt(np.power(tr[0] - tr[0], 2) + np.power(tr[1] - tl[1], 2))
    maxWidth = max(widthA, widthB)
    
    heightA = np.sqrt(np.power(tr[0] - br[0], 2) + np.power(tr[1] - br[1], 2))
    heightB = np.sqrt(np.power(tl[0] - bl[0], 2) + np.power(tl[1] - bl[1], 2))
    maxHeight = max(heightA, heightB)
    
    dst = np.array([
        [0,0], 
        [maxWidth - 1, 0], 
        [maxWidth - 1, maxHeight -1],
        [0, maxHeight - 1]], dtype = np.float32)
    
    M = cv2.getPerspectiveTransform(rect, dst)
    #print(M)
    #print(maxWidth, maxHeight)
    warped = cv2.warpPerspective(img, M, (int(maxWidth), int(maxHeight)))
    return warped

warped = four_point_transform(img, mainEdge)

# save the warped image, but first resize to width = 500

#cv2.imshow("Warped", warped)
(h, w) = warped.shape[:2]
r = 500/float(w)
dim = (500, int(r*h))
warped = cv2.resize(warped, dim, interpolation=cv2.INTER_AREA)
cv2.imwrite("warped_img.png", warped)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

