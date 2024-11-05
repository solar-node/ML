import cv2
from PIL import Image # importing Image function from  PILLOW library
import numpy as np


# Function to calculate the HSV color range (lower and upper limits) for a given BGR color
def get_limits(color):
    # Convert the BGR color (input) to a numpy array for OpenCV's color conversion function
    c = np.uint8([[color]])  # Wrap color in extra brackets to fit the expected input shape for conversion
    
    # Convert the BGR color to HSV color space to isolate hue
    hsvColor = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    # Define the lower and upper HSV range
    # Lower limit: Subtract 10 from the hue to cover a range of shades, and set saturation & value to minimum (100)
    lowerLimit = hsvColor[0][0][0] - 10, 100, 100

    # Upper limit: Add 10 to the hue, and set saturation & value to maximum (255)
    upperLimit = hsvColor[0][0][0] + 10, 255, 255

    # Convert the lower and upper limits to numpy arrays with data type uint8 (8-bit unsigned integers)
    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    # Return the lower and upper limits for the color range in HSV format
    return lowerLimit, upperLimit



yellow = [255, 0,0]  #Put thhese colors in BGR

cap = cv2.VideoCapture(0)   # No. depends on how many webcams are in you system; (0) for default cam


# capturing video   and showing it frame by  frame
while True:
    ret, frame = cap.read()

    # Converting the image from BGR to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # get_limits will  return  the limits of the yellow color
    lowerLimit, upperLimit = get_limits(color=yellow)


    #  We will apply inRange function :Use the function to create a mask that highlights only the colors within this range.
    mask = cv2.inRange(hsvImage,lowerLimit, upperLimit)

    # Converting the image of mask (which is a numpy  array  into  pillow format)
    mask2 = Image.fromarray(mask)

    # Getting boundry box from the pillow using getbbox
    bbox = mask2.getbbox()  # function to get boundry  box around the object
    # print(bbox)   # to  confirm is bbox working or not 

    if bbox is not None:
        x1, y1, x2, y2 = bbox   #format of bbox : (1765, 574, 1769, 575)

        frame = cv2.rectangle(frame, (x1, y1),(x2, y2), (0,255, 0), 5 )
        
        # Put the text on the image
        cv2.putText(frame, "Yellow", (x1, y1-10), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 0), 2)
    


    cv2.imshow('frame', frame)  #Shows the captured video as image frame by  frame


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()   #Releasing the memory
cv2.destroyAllWindows()

 