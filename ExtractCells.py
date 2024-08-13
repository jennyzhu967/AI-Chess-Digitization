import os
import cv2
import pyautogui
import numpy as np
from torchvision.transforms.functional import autocontrast
global click_list 

image_path = "C:/Users/ericz/Mouse Clicker/Games/StephenSophie.png"
image = cv2.imread(image_path)

screen_width, screen_height = pyautogui.size()
# print(f"Screen Width: {screen_width}, Screen Height: {screen_height}")
image_height, image_width, color_channels = image.shape
# print(f"Original Height: {image_height}, Original Width: {image_width}, Original Channels: {color_channels}\n")

resize_metric = 0.8
while True:
    if (image_height < screen_height and image_width < screen_width):
        break

    image_width = int(image_width * resize_metric)
    image_height = int(image_height * resize_metric)
    # print("Resizing image to:  ({}, {})\n".format(image_width, image_height))

    image = cv2.resize(image,(image_width, image_height))

click_list =[]
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ',', y) 
        # cv2.circle(image,(x,y),5,(255,0,0),-1)
        click_list.append([x,y])

cv2.imshow("image", image)
cv2.setMouseCallback("image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# hard coded click_list
# click_list = [[112 , 310], [246 , 310], [250 , 967], [116 , 971]]

def bottom_corner(x1, y1, x2, y2, factor):
    list = []
    absolute_x_difference = abs(x1-x2) # look into this
    per_x_unit = (absolute_x_difference / 30) * factor
    
    absolute_y_difference = abs(y1-y2)
    per_y_unit = (absolute_y_difference / 30) * factor

    list.append(x1 - per_x_unit)
    list.append(y1 + per_y_unit)

    return list

# constants (these will not change)
H_SIZE = 30
cnt = 0 # for seeing x images
page = "0"  

gameNumber = "004" # change this for different games
player = "white" # change for white and black
move = 1 # change to 30 when different row

topLeft = [click_list[0][0], click_list[0][1]] # type list for the upper left x and y points
topRight = [click_list[1][0], click_list[1][1]] # upper right x and y points in a list [x, y]
bottomRight = [click_list[2][0], click_list[2][1]]
bottomLeft = [click_list[3][0], click_list[3][1]]

previousLeft = [topLeft]
previousRight = [topRight]

ratioFactor = 1

for ih in range(0,H_SIZE):
    # calculate coordinates for bottom left and right
    bottomLeftPoints = [bottom_corner(topLeft[0], topLeft[1], bottomLeft[0], bottomLeft[1], ratioFactor)]
    bottomRightPoints = [bottom_corner(topRight[0], topRight[1], bottomRight[0], bottomRight[1], ratioFactor)]
    
    points = [bottomLeftPoints, previousLeft, previousRight, bottomRightPoints]
    points = np.array(points)
    points = points.astype(int)

    rect = cv2.minAreaRect(points) # draw the minumum rectangle possible with these points
    box = cv2.boxPoints(rect) # get the coordinates
    # returns lowest point and goes clockwise
    box = np.intp(box)
    
    new_image = cv2.drawContours(image, [box], 0, (0,0,255),1) # contour the box

    # show images
    if cnt < 40:
        cv2.imshow("cropped image",new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    name = gameNumber + "_" + page + "_" + str(move) + "_" +player+ ".png"
    directory = "C:/Users/ericz/Mouse Clicker/test"
    SAVED_PATH = os.path.join(directory, name).replace("\\", "/")
    
    move+=1
    ratioFactor+=1
    previousLeft = bottomLeftPoints
    previousRight = bottomRightPoints
    
    if cnt == 29: # last point
        print(f"We ended bottom Left: {bottomLeftPoints}")
        print(f"Theretical bottom Left: {click_list[3]} \n")

        print(f"We ended bottom Right: {bottomRightPoints}")
        print(f"Theretical bottom Right: {click_list[2]}")
    
    cnt+=1