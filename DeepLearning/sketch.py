import pygame
import numpy as np
import cv2

#Training data
device = "gpu"
model_filename = "PyplotGraphs/Model.h5"

#Keras
import os
os.environ['THEANORC'] = "./" + device + ".theanorc"
from keras.models import Sequential, load_model
from keras import backend as K
K.set_image_data_format('channels_first')

#Display variables
border_offset = 10
image_scale = 2
input_width = 200
input_height = 200
subwindow_width = input_width * image_scale #int((window_width / 2) - (border_offset * 1.5))
subwindow_height = input_height * image_scale #int(window_height - (border_offset * 2))
background_colour = (222, 172, 199)
window_width = subwindow_width * 2 + border_offset * 3
window_height = subwindow_height + border_offset * 2

input_x = border_offset
input_y = border_offset
output_x = input_x + subwindow_width + border_offset
output_y = input_y

#Mouse variables
previous_mouse_pos = None
mouse_pressed = False
should_update = True
current_colour = 1
current_sketch = None
current_output = np.zeros((3, input_height, input_width), dtype=np.uint8)
#How smooth the line drawn is
mouse_interpolation = 100
rgb_array = np.zeros((input_height, input_width, 3), dtype=np.uint8)
img_result = np.zeros((input_height, input_width, 3), dtype=np.uint8)

#Load the model
model = load_model(model_filename)

#Open a new pygame window
pygame.init()
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Building LOD Generator - Ellie Dunstan')

#Define the windows and add them to the main window
input_window_surface = pygame.Surface((subwindow_width, subwindow_height))
input_window = window.subsurface((input_x, 
                                  input_y, 
                                  subwindow_width, 
                                  subwindow_height))
output_window_surface = pygame.Surface((subwindow_width, subwindow_height))
output_window = window.subsurface((output_x, 
                                   output_y, 
                                   subwindow_width, 
                                   subwindow_height))

def add_pos(arr):
	s = arr.shape
	result = np.empty((s[0], s[1] + 2, s[2], s[3]), dtype=np.float32)
	result[:,:s[1],:,:] = arr
	x = np.repeat(np.expand_dims(np.arange(s[3]) / float(s[3]), axis=0), s[2], axis=0)
	y = np.repeat(np.expand_dims(np.arange(s[2]) / float(s[2]), axis=0), s[3], axis=0)
	result[:,s[1] + 0,:,:] = x
	result[:,s[1] + 1,:,:] = np.transpose(y)
	return result

def clear_sketch():
    global current_sketch
    #array filled with zeros
    current_sketch = np.zeros((3,subwindow_height, subwindow_width), dtype=np.uint8) 

clear_sketch()

def on_mouse_click(mouse_pos):
    global current_colour
    #global current_sketch
    global should_update
    #global image_scale
    #else position is at bottom right of mouse cursor
    #mouse_offset = 10
    x = (mouse_pos[0] - output_x)# / image_scale
    y = (mouse_pos[1] - output_y)# / image_scale
    #if mouse is not in the drawing box
    if not (x >= 0 and y >= 0 and x < subwindow_width and y < subwindow_height):
        x = mouse_pos[0] - input_x#) / image_scale)
        y = mouse_pos[1] - input_y#) / image_scale)
    #if mouse is in drawing box
    if x >= 0 and y >= 0 and x < subwindow_width and y < subwindow_height:
        should_update = True
        current_sketch[0, y, x] = 255


def on_mouse_move(mouse_pos):
    global previous_mouse_pos
    if previous_mouse_pos is None:
        previous_mouse_pos = mouse_pos
    if current_colour == 1:
        for i in range(0, mouse_interpolation):
            a = float(i) / mouse_interpolation
            x = int((1.0 - a) * mouse_pos[0] + a * previous_mouse_pos[0])
            y = int((1.0 - a) * mouse_pos[1] + a * previous_mouse_pos[1])
            on_mouse_click((x, y))
    else:
        on_mouse_click(mouse_pos)
    previous_mouse_pos = mouse_pos


def sparse_to_rgb(sparse_arr):
    t = np.repeat(sparse_arr, 1, axis=0)
    return np.transpose(t, (2, 1, 0))


def draw_sketch():
    pygame.surfarray.blit_array(input_window_surface, rgb_array)
    pygame.transform.scale(input_window_surface, (subwindow_width, subwindow_height), input_window)
    pygame.draw.rect(window, (83, 83, 255), (input_x, input_y, subwindow_width, subwindow_height), 1)


def draw_output():
    pygame.surfarray.blit_array(output_window_surface, np.transpose(current_output, (2, 1, 0)))
    pygame.transform.scale(output_window_surface, (subwindow_width, subwindow_height), output_window)
    pygame.draw.rect(window, (0, 0, 0), (output_x, output_y, subwindow_width, subwindow_height), 1)

num_images_saved = 0

def save_image():
    global num_images_saved
    pygame.image.save(output_window_surface, "Sketches/image" + str(num_images_saved) + ".png")
    num_images_saved += 1

#Main loop
running = True
while running:
    #mouse-related events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                previous_mouse_pos = pygame.mouse.get_pos()
                on_mouse_click(previous_mouse_pos)
                mouse_pressed = True
            elif pygame.mouse.get_pressed()[2]:
                clear_sketch()
                should_update = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False
            previous_mouse_pos = None
        elif event.type == pygame.MOUSEMOTION and mouse_pressed:
            on_mouse_move(pygame.mouse.get_pos())
        elif pygame.key.get_pressed()[pygame.K_s]:
            save_image()

    #update drawing and prediction
    #if should_update:
    drawing = np.expand_dims(current_sketch.astype(np.float32) / 255.0, axis = 0)
    prediction = model.predict(add_pos(drawing), batch_size = 1)[0]
    current_output = (prediction * 255.0).astype(np.uint8)
    rgb_array = sparse_to_rgb(current_sketch)
    should_update = False
        
    window.fill(background_colour)
    draw_sketch()
    draw_output()

    pygame.display.flip()
    pygame.time.wait(10)