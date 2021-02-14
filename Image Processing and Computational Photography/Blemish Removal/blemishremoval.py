# -*- coding: utf-8 -*-
"""
Submission for blemish removal
"""
def identifyBestPatch(x, y, r):
    patches = {}
    # Find the gradient for the 8 neighbors
    keyTup = appendDict(x+2*r, y)
    patches['Right'] = (x + 2*r, y, keyTup[0], keyTup[1])
    
    keyTup = appendDict(x+2*r, y+2*r)
    patches['BotRight'] = (x + 2*r, y + r, keyTup[0], keyTup[1])

    keyTup = appendDict(x, y+2*r)
    patches['Bot'] = (x, y + 2*r, keyTup[0], keyTup[1])
    
    keyTup = appendDict(x-2*r, y+2*r)
    patches['BotLeft'] = (x - 2*r, y + 2*r, keyTup[0], keyTup[1])
    
    keyTup = appendDict(x-2*r, y)
    patches['Left'] = (x - 2*r, y, keyTup[0], keyTup[1])
    
    keyTup = appendDict(x-2*r, y-2*r)
    patches['TopLeft'] = (x - 2*r, y - 2*r, keyTup[0], keyTup[1])
    
    keyTup = appendDict(x, y-2*r)
    patches['Top'] = (x, y - 2*r, keyTup[0], keyTup[1])

    keyTup = appendDict(x+2*r, y-2*r)
    patches['TopRight'] = (x + 2*r, y - 2*r, keyTup[0], keyTup[1])
    
    #print(patches)
    # Find the direction with the lowest magnitude of gradient
    findlow_grad = {}
    for key, (x, y, gx, gy) in patches.items():
        findlow_grad[key] = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
    #print(findlow_grad)
    
    key_min = min(findlow_grad.keys(), key=(lambda k: findlow_grad[k]))
    #print(key_min)
    return patches[key_min][0], patches[key_min][1]
                   
def sobelfilter(crop_img):
    # use Sobel to find the first order derivative
    sobel_x = cv2.Sobel(crop_img, cv2.CV_32F, 1, 0, ksize=3) # x
    abs_sobel_x = np.absolute(sobel_x)
    sobel_x8u = np.uint8(abs_sobel_x)
    grad_x = np.mean(sobel_x8u)

    sobel_y = cv2.Sobel(crop_img, cv2.CV_32F, 0, 1, ksize=3) # y
    abs_sobel_y = np.absolute(sobel_y)
    sobel_y8u = np.uint8(abs_sobel_y)
    grad_y = np.mean(sobel_y8u)    
    return grad_x, grad_y

def appendDict(x,y):
    crop_img = source[y-r:y+r, x-r:x+r]
    grad_x, grad_y = sobelfilter(crop_img)
    return grad_x, grad_y  

def onMouse(action, x, y, flags, userdata):
    global r, source, windowName

    if action == cv2.EVENT_LBUTTONUP:
        # TODO: check for boundary conditions
        loc = (x, y)
        # find the best patch to replace the blemish
        newX, newY = identifyBestPatch(x, y, r)
        newPatch = source[newY-r: newY+r, newX-r: newX+r]
        
        mask = 255*np.ones(newPatch.shape, newPatch.dtype)
        # use seamlessClone to blend the new patch
        source = cv2.seamlessClone(newPatch, source, mask, loc, cv2.NORMAL_CLONE)
        # show the output
        cv2.imshow(windowName, source)

if __name__ == '__main__':
    import cv2
    import numpy as np
    r = 15
    windowName = 'Blemish Removal'
    source = cv2.imread('blemish.png', cv2.IMREAD_COLOR)
    dummy = source.copy()
    
    print('Using a patch of radius 15')
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, onMouse)
    
    k = 0
    # loop until esc
    while k != 27:
        cv2.imshow(windowName, source)
        k = cv2.waitKey(20) & 0xFF
        if k == 99: # press 'c' to restart
            source = dummy.copy()
    cv2.destroyAllWindows()
