import numpy as np
import cv2
import glob

from keras.preprocessing.image import img_to_array

colour_images = []
lined_images = []
image_dir = "Dataset/*RESIZED.jpg"

files = glob.glob(image_dir)
print ("Reading images...")
for file in files:
    img = cv2.imread(file)
    if "lines" in file:
        lined_images.append(img)
    else:
        colour_images.append(img)

num_images = len(lined_images)
image_dims = 400
samples_per_image = 10
num_samples = num_images * 2 * samples_per_image

#print ("Coloured images length: " + str(len(colour_images)))
#print ("Lined images length: " + str(len(lined_images)))
#print ("Num samples " + str(num_samples))

def generate_numpy_arrays():
    #3 refers to channels i.e. RGB
    lined_data = np.empty((num_images, 3, image_dims, image_dims), dtype=np.uint8)
    coloured_data = np.empty((num_images, 3, image_dims, image_dims), dtype=np.uint8)

    for iter in range(0, num_images):
        print ("images processed: " + str(iter))
        #rearrange from 600, 600, 3 to 3, 600, 600
        img = np.transpose(lined_images[iter], (2, 0, 1))
        lined_data[iter] = np.flip(img, axis = 2)
        img = np.transpose(colour_images[iter], (2, 0, 1))
        coloured_data[iter] = np.flip(img, axis = 2)

    print ("Saving numpy files...")
    np.save("x_data.npy", lined_data)
    np.save("y_data.npy", coloured_data)

generate_numpy_arrays()