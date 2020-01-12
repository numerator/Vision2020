# read and process an image

import cv2
import numpy as np

displayImages =  True

def removeNoise(hsv_img, kernelSize, lower_color_range, upper_color_range):
    # Kernal to use for removing noise
    kernel = np.ones(kernelSize, np.uint8)
    # Convert image to binary
    mask = cv2.inRange(hsv_img, lower_color_range, upper_color_range)
    # Show the binary (masked) image
    # if(displayImages):
    #     cv2.imshow("img", mask)
    # Close the gaps (due to noise) in the masked image
    close_gaps = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Remove noisy parts of the masked image
    no_noise = cv2.morphologyEx(close_gaps, cv2.MORPH_OPEN, kernel)
    # Undo the erosion to the actual target done during noise removal
    dilate = cv2.dilate(no_noise, np.ones((5,10), np.uint8), iterations=5)
    return dilate

def findObjectContours(dilate, objName):
    # Find boundary of object
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoured_image = cv2.drawContours(dilate, contours, -1, (0, 0, 255), 2)
    cv2.imshow("Contours", contoured_image)
    cv2.waitKey(5000)
    # # Only proceed if contours were found
    # if(contours != None):
    #     if(len(contours) > 1):
    #         sorted(contours, key=lambda contour: getApproximateArea(contour), reverse=True)
    #         contour_boundaries = []
    #         if True: # (len(contours) < 4):
    #             contour_boundaries = [getContourBoundary(contours[0]), getContourBoundary(contours[1])]
    #         else:
    #             interesting_contours = contours[:4]
    #             sorted(interesting_contours, key=lambda contour: abs(getCenterPoint(getContourBoundary(contour))[0] - frame_width/2))
    #             contour_boundaries = [getContourBoundary(interesting_contours[0]), getContourBoundary(interesting_contours[1])]
    #             # TODO: Add code to threshold area of contours
    #             # print("Interesting contours type:", type(interesting_contours))
    #         if sendPackets:
    #             prepareForRoboRIO(contour_boundaries, objName)
    #         else:
    #             sendData(False, 0, 0, "")
    #         for contour_boundary in contour_boundaries[:-1]:
    #             displayObject(contour_boundary, objName)
    #         return displayObject(contour_boundaries[-1], objName)

if __name__ == "__main__":

    bgr_img = cv2.imread("test_images/whiteboard_hex.png")
    # cv2.imshow("Hello", bgr_img)
    # cv2.waitKey(5000) # keeps the window open for 5 seconds--long enough to load the image

    # # Convert the frame to HSV
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # # cv2.imshow("Win1", bgr_img)

    # # Find the cube
    hex_hsv_lower = np.array([80, 30, 0])
    hex_hsv_upper = np.array([255, 255, 60])
    hex_dilate = removeNoise(bgr_img, (5,5), hex_hsv_lower, hex_hsv_upper)
    # #cv2.imshow("dilated image", hex_dilate)
    # print (hex_dilate.shape)

    hex_img = np.array([])
    hex_img = findObjectContours(hex_dilate, "retroreflective")

    # # # Display the BGR image with found objects bounded by rectangles
    # # if(displayImages):
    # #     cv2.imshow("Objects found!", bgr_img)

