import cv2
import numpy as np
import glob
import os
import sys
import time
import json
# from networktables import NetworkTables
# from networktables.util import ntproperty


def removeNoise(hsv_img, kernelSize, lower_color_range, upper_color_range):
    # Kernal to use for removing noise
    kernel = np.ones(kernelSize, np.uint8)
    # Convert image to binary
    mask = cv2.inRange(hsv_img, lower_color_range, upper_color_range)
    # Show the binary (masked) image
    if(displayImages):
        cv2.imshow("img", mask)
    # Close the gaps (due to noise) in the masked image
    close_gaps = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Remove noisy parts of the masked image
    no_noise = cv2.morphologyEx(close_gaps, cv2.MORPH_OPEN, kernel)
    # Undo the erosion to the actual target done during noise removal
    dilate = cv2.dilate(no_noise, np.ones((5,10), np.uint8), iterations=5)
    return dilate

def getCenterPoint(top_left, bottom_right):
    return (int((top_left[0]+bottom_right[0])/2), int((top_left[1]+bottom_right[1])/2))

def displayObject(contour_boundary, objName):
    # If the object is cube, use red, if retroreflective, use blue
    if(objName == "cube"):
        color = (0,0,255)
    elif(objName == "retroreflective"):
        color = (255,0,0)

    top_left, bottom_right = contour_boundary

    # Draw a rectangle bounding the object using top left and bottom right points
    cv2.rectangle(bgr_img, top_left, bottom_right, color, 3)
    # Find the center point of the object
    center_point = getCenterPoint(top_left, bottom_right)

    # Draw circle at the center point
    cv2.circle(bgr_img, center_point, 5, color, -1)

    # isOffCenter = False # Enable if camera is off center
    # angle = None
    # if(isOffCenter): # If camera is NOT in center, use this
    #     # Find the angle to the center point
    #     offset = 90 # Change this as needed
    #     adjusted_point = center_point[0] - offset
    #     angle = getAngle(adjusted_point)
    #     cv2.circle(bgr_img, (adjusted_point, center_point[1]), 5, color, -1)
    # else:
    #     # If camera IS in center, use this
    #     angle = getAngle(center_point[0])
    # print(objName, ":", angle)
    # If the program isn't in testing mode, send data to RoboRIO
    # if(sendPackets):
    #     sendData(angle, width, objName)
    return hsv_img

def getContourBoundary(contour):
    # Extract boundary points of object
    left = tuple(contour[contour[:,:,0].argmin()][0])
    right = tuple(contour[contour[:,:,0].argmax()][0])
    top = tuple(contour[contour[:,:,1].argmin()][0])
    bottom = tuple(contour[contour[:,:,1].argmax()][0])

    # Use boundary points to find the top left and bottom right corners
    top_left = (left[0], top[1])
    bottom_right = (right[0], bottom[1])

    return top_left, bottom_right

def getApproximateArea(contour):
    top_left, bottom_right = getContourBoundary(contour)
    return abs((bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1]))

def prepareForRoboRIO(contour_boundaries, objName):
    contour_boundary_1, contour_boundary_2 = contour_boundaries

    angle = getAngle(getCenterPoint(contour_boundary_1[0], contour_boundary_2[1])[0])
    width = abs(contour_boundary_1[0][0] - contour_boundary_2[0][0]) # TODO : Check if subscripting is correct

    sendData(True, angle, width, objName)

def findObjectContours(dilate, objName):
    # Find boundary of object
    _, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Only proceed if contours were found
    if(contours != None):
        if(len(contours) > 1):
            sorted(contours, key=lambda contour: getApproximateArea(contour), reverse=True)
            contour_boundaries = []
            if True: # (len(contours) < 4):
                contour_boundaries = [getContourBoundary(contours[0]), getContourBoundary(contours[1])]
            else:
                interesting_contours = contours[:4]
                sorted(interesting_contours, key=lambda contour: abs(getCenterPoint(getContourBoundary(contour))[0] - frame_width/2))
                contour_boundaries = [getContourBoundary(interesting_contours[0]), getContourBoundary(interesting_contours[1])]
                # TODO: Add code to threshold area of contours
                # print("Interesting contours type:", type(interesting_contours))
            if sendPackets:
                prepareForRoboRIO(contour_boundaries, objName)
            else:
                sendData(False, 0, 0, "")
            for contour_boundary in contour_boundaries[:-1]:
                displayObject(contour_boundary, objName)
            return displayObject(contour_boundaries[-1], objName)

def getAngle(point):
    # Use the center_point, fov, and width to find the heading (angle to target)
    field_of_view = 65
    pixel_distance = point - frame_width/2
    heading = ((field_of_view/2.0) * pixel_distance)/(frame_width/2)
    print("-"*int(heading+24) + " * " + "-"*int(24-heading))
    return int(heading)

def sendData(status, angle, width, objName):
    # Put the data (to be sent to the RIO) in a dictionary
    data = {
        "status" : status,
        "sender" : "vision",
        "object" : objName,
        "angle" : int(angle),
        "width" : int(width),
        "id" : counter
    }
    reporter.vision_bearing = angle

def reduceExposure():
    zero = "v4l2-ctl --device=/dev/video0 -c gain_automatic=0 -c "
    one = "white_balance_automatic=0 -c exposure=5 -c gain=0 -c "
    two = "auto_exposure=1 -c brightness=0 -c hue=-32 -c saturation=96"
    cmd0 = zero + one + two
    print("cmd:", cmd0)
    os.system(cmd0)
    zero = "v4l2-ctl --device=/dev/video0 -c gain_automatic=0 -c "
    one = "white_balance_automatic=0 -c exposure=5"
    cmd1 = zero + one
    print(cmd1)
    os.system(cmd1)

class VisionReporter(object):
    vision_bearing = ntproperty("/SmartDashboard/vision-bearing", 0.0)

# while True:
#     result = NetworkTables.initialize(server = '10.10.76.2')
#     if not result:
#         break
# reporter = VisionReporter()

# Set up a counter, for use in logging images
counter = 0
# Track if the program has ran (if not, create a new folder for image logging)
ranOnce = False
# Folder path for logging images
folder = ""
# Track if the program is being tested
isTesting = False
# Should the images be displayed on screen?
displayImages = False
# Should packets be sent?
sendPackets = True
# Should exposure be reduced?
shouldReduceExposure = True
# If test is found in the cmd line arguments, then the program is testing
for arg in sys.argv:
    if(not isTesting):
        if(arg == "cube"):
            shouldReduceExposure = False
        if(arg == "test"):
            # When testing, use an alternate filepath
            folder = "/Users/cbmonk/Downloads/ImageLogging/"
            isTesting = True
        else:
            folder = "/var/log/Vision"
            if(shouldReduceExposure):
            	reduceExposure()
    if(arg == "displayimages"):
        displayImages = True
    if(arg == "nopackets"):
        sendPackets = False

# Set up the webcam input
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
video_capture.set(cv2.CAP_PROP_FPS, 10)


# find the resolution of the webcam input
_, bgr_img = video_capture.read()
_, frame_width, _ = bgr_img.shape
print("----Width: " + str(frame_width)+" Fps:" + str(video_capture.get(cv2.CAP_PROP_FPS)))

def logImage(bgr_img, folder, ranOnce):
    # Default index to use if no previous logging folders exist
    logging_folder = "0001"
    # Change the current directory to the logging folder (defined before this for loop began)
    os.chdir(folder)
    sorted_glob = sorted(glob.glob("[0-9][0-9][0-9][0-9]"))
    path = os.path.join(folder, str(counter) +".jpg")
    if len(sorted_glob)>0 and (not ranOnce):
        # Make a new folder with a 4 digit name one greater than the last logging folder
        logging_folder = "{:04d}".format(int(sorted_glob[-1])+1)
        # print(logging_folder)
        # If this is the first time the program has been run, make a logging folder
        os.mkdir(logging_folder)
        # Path for the image to be saved
        path = os.path.join(folder, logging_folder, str(counter) + ".jpg")
    # print(path)
    # Save the image
    cv2.imwrite(path, bgr_img)

    if(not ranOnce):
        # Return the filepath so it can be stored outside this scope
        return os.path.join(folder, logging_folder)
    return folder
if __name__ == "__main__":
    while(True):
        time0 = time.time()
        # Read the frame from the video capture
        _, bgr_img = video_capture.read()
        bgr_img = cv2.imread("test_images/whiteboard_hex.png")
        # Convert the frame to HSV
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        # Enable line below if reading from precaptured image
        # bgr_img = cv2.imread("/Users/cbmonk/Downloads/testf.png")

        # Find the cube
        cube_hsv_lower = np.array([25, 100, 100])
        cube_hsv_upper = np.array([28, 255, 215])
        # cube_dilate = removeNoise(hsv_img, (5,5), cube_hsv_lower,cube_hsv_upper)

        # Find the retroreflective tape
        # Use these HSV values if the LEDs are very bright and exposure is normal
        # retro_hsv_lower = np.array([0, 0, 255])
        # retro_hsv_upper = np.array([0, 0, 255])
        # Enable the below values if LEDs are NOT bright enough
        # retro_hsv_lower = np.array([87, 155, 230])
        # retro_hsv_upper = np.array([95, 200, 255])
        retro_hsv_lower = np.array([40, 210, 55]) # np.array([43, 125, 171])
        retro_hsv_upper = np.array([55, 255, 255])
        retro_dilate = removeNoise(hsv_img, (5,5), retro_hsv_lower, retro_hsv_upper)
        retro_img = np.array([])
        retro_img = findObjectContours(retro_dilate, "retroreflective")

        # Display the BGR image with found objects bounded by rectangles
        if(displayImages):
            cv2.imshow("Objects found!", bgr_img)

        # Log every 10th BGR image with the bounding boxes displayed
        if(counter%10 == 0):
            folder = logImage(bgr_img, folder, ranOnce)
            ranOnce = True

        # Keep track of how many times the program has run (for image logging)
        counter+=1
        ranOnce = True
        print("Time:", time.time()-time0)

# Release the video capture and close the windows when q is pressed
video_capture.release()
cv2.destroyAllWindows()
