import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib

#from dataPath import DATA_PATH
#from dataPath import MODEL_PATH
DATA_PATH = '' # NOTE: I have uploaded the model files, and high_contrast.jpg file that I used
MODEL_PATH = ''

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0,8.0)
matplotlib.rcParams['image.cmap'] = 'gray'

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def detect_face_landmark(faceCascade, predictor, imGray):
    # Detect face using HaarCascade
    faces = faceCascade.detectMultiScale(
        imGray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
        )
    (x, y, w, h) = faces[0] # assume a single face in imGray
    # Facial landmark using dlib
    # https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    shape = predictor(imGray, dlib.rectangle(x, y, x+w, y+h))
    shape = shape_to_np(shape)
    return shape

# image of face
im = cv2.imread(DATA_PATH + 'musk.jpg')
imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# Make a copy
faceImg = im.copy()

cascPath = MODEL_PATH + 'haarcascade_frontalface_default.xml'
predPath = MODEL_PATH + 'shape_predictor_68_face_landmarks.dat'

faceCascade = cv2.CascadeClassifier(cascPath)
predictor = dlib.shape_predictor(predPath)

shape = detect_face_landmark(faceCascade, predictor, imGray)

# width of face
width_of_face = shape[16][0] - shape[0][0]
border = int(0.1*width_of_face) # add border to the face

width_of_glasses = width_of_face + border

# glasses
glassPNG = cv2.imread(DATA_PATH + "images/sunglass.png", -1)

# Resize the image to fit over the eye region
(h, w) = glassPNG.shape[:2]
r = width_of_glasses / float(w)

glassPNG = cv2.resize(glassPNG,(width_of_glasses, int(h * r)))
glassBGR = glassPNG[...,0:3] # BGR
glassMask1 = glassPNG[:,:,3] # alpha channel

# High contrast image
high_cont_img = cv2.imread('high_contrast.jpg')
(h, w) = high_cont_img.shape[:2]
r = width_of_glasses / float(w)

# Resize the image to make it the same width as that of glasses
high_cont_img = cv2.resize(high_cont_img,(width_of_glasses, int(h * r)))
# Crop the image to make it the same height as that of glasses
high_cont_img = high_cont_img[0:glassMask1.shape[0],0:glassMask1.shape[1]]

# Convert to gray scale
high_cont_img = cv2.cvtColor(high_cont_img, cv2.COLOR_BGR2GRAY)

# Erode the glassMask1 to remove some of the thin lines
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
glassMaskMain = cv2.erode(glassMask1, kernel, iterations = 2)

# Make the dimensions of the mask same as the input image.
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMaskA = np.uint8(glassMask/255)
# Glass image
maskedGlass = cv2.multiply(glassBGR, glassMaskA)
# Mask of the high contrast img
maskedImg = cv2.multiply(high_cont_img, np.uint8(glassMaskE/255))
maskedImg = cv2.merge((maskedImg, maskedImg, maskedImg))

# Mask of the final image, alpha = 0.6, beta = 0.4
maskedFinal = cv2.addWeighted(maskedGlass, 0.6, maskedImg, 0.4, 0.0)

# Get the eye region from the face image
ys = shape[24][1]
xs = shape[0][0] - int(border/2)
eyeROI = faceImg[ys:ys+glassMaskB.shape[0],xs:xs+width_of_glasses]

# Transparent glasses to show the eyes
glassMaskB = np.uint8(glassMask*0.4)
maskedEye = cv2.subtract(eyeROI, glassMaskB)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedFinal)

# Replace the eye ROI with the output
faceImg[ys:ys+glassMaskB.shape[0],xs:xs+width_of_glasses]=eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(im[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceImg[:,:,::-1]);plt.title("With Sunglasses");