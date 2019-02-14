import numpy as np
#Grab file paths
import glob
import os
import cv2
import argparse
from PIL import Image


#automatic line detection
#lower value of sigma indicates tighter threshold
def auto_canny(image, sigma=0.33):
    #median of the pixel intensities in the image
    v = np.median(image)

    #apply auto canny edge detection
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    #return edged image
    return edged


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--Dataset", required = True,
                help = "path to input dataset of images")
args = vars(ap.parse_args())



def create_lined_images():
    #loop through images
    for imagePath in glob.glob(args["Dataset"] + "/*.jpg"):
    
        if imagePath.__contains__("lines"):
            continue
    
        image = cv2.imread(imagePath)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    
        #wide = cv2.Canny(blurred, 10, 200)
        #tight = cv2.Canny(blurred, 225, 250)
        auto = auto_canny(blurred)
    
        oldPath = imagePath + "lines.jpg"
        linedPath = imagePath[:-4] + "lines.jpg"
    
        if glob.glob(linedPath):
            os.remove(linedPath)
        if glob.glob(oldPath):
            os.remove(oldPath)
    
        cv2.imwrite(linedPath, auto)
        print ("image processed ", linedPath)
    
        #display images
        #cv2.imshow("Original", image)
        #cv2.imshow("Edges", np.hstack([wide, tight, auto]))
        #cv2.waitKey(0)

def resize_images():
    image_dimensions = 600
    #loop through images
    for imagePath in glob.glob(args["Dataset"] + "/*.jpg"):
        img = Image.open(imagePath)
        img = img.resize((image_dimensions, image_dimensions))
        img.save(imagePath[:-4] + "RESIZED.jpg")



#create_lined_images()
resize_images()