import cv2
import math

# Lists to store the top left corner point
topleft=[]

def drawRectangle(action, x, y, flags, userdata):
  # Referencing global variables 
  global topleft
  # Action to be taken when left mouse button is pressed
  if action==cv2.EVENT_LBUTTONDOWN:
    topleft=[(x,y)]

  # Action to be taken when left mouse button is released
  elif action==cv2.EVENT_LBUTTONUP:
    # Draw the rectangle
    faceImg = dummy[topleft[0][1]:y, topleft[0][0]:x]
    cv2.imwrite("face.png", faceImg)

    cv2.rectangle(source, topleft[0], (x,y), (255,0,255),thickness=2, 
                    lineType=cv2.LINE_AA)
    cv2.imshow("Window",source)


source = cv2.imread("sample.jpg", cv2.IMREAD_COLOR)
# Make a dummy image, will be useful to clear the drawing
dummy = source.copy()
cv2.namedWindow("Window")
# highgui function called when mouse events occur
cv2.setMouseCallback("Window", drawRectangle)
k = 0
# loop until escape character is pressed
while k!=27 :
  
  cv2.imshow("Window", source)
  cv2.putText(source,'''Choose top left corner, and drag''', 
              (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
              0.7,(255,255,255), 2 );
  k = cv2.waitKey(20) & 0xFF
  # Another way of cloning
  if k==99:
    source = dummy.copy()

cv2.destroyAllWindows()
