import pygame
import numpy as np

#Display variables
window_width = 1280
window_height = 720
border_offset = 10
subwindow_width = int((window_width / 2) - (border_offset * 1.5))
subwindow_height = int(window_height - (border_offset * 2))
background_colour = (222, 172, 199)

#Mouse variables
previous_mouse_pos = None
mouse_pressed = False
should_update = True
current_colour = 1
current_sketch = None
mouse_interpolation = 10
rgb_array = np.zeros((subwindow_height, subwindow_width, 3), dtype=np.uint8)

#Open a new pygame window
pygame.init()
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Building LOD Generator - Ellie Dunstan')

#Define the windows and add them to the main window
input_window_surface = pygame.Surface((subwindow_width, subwindow_height))
input_window = window.subsurface((border_offset, 
                                  border_offset, 
                                  subwindow_width, 
                                  subwindow_height))
output_window_surface = pygame.Surface((subwindow_width, subwindow_height))
output_window = window.subsurface((subwindow_width, 
                                   border_offset, 
                                   subwindow_width, 
                                   subwindow_height))


def clear_sketch():
    global current_sketch
    #array filled with zeros
    current_sketch = np.zeros((1,subwindow_height, subwindow_width), dtype=np.uint8) 


clear_sketch()


def on_mouse_click(mouse_pos):
    global current_colour
    global should_update
    #else position is at bottom right of mouse cursor
    mouse_offset = 10
    x = mouse_pos[0] - mouse_offset
    y = mouse_pos[1] - mouse_offset
    #if mouse in drawing box
    if (x >= 0 and y >= 0 and x < subwindow_width and y < subwindow_height):
        should_update = True
        current_sketch[0, y, x] = 255


def on_mouse_move(mouse_pos):
    global previous_mouse_pos
    if previous_mouse_pos is None:
        previous_mouse_pos = mouse_pos
    if current_colour == 1:
        for i in range(mouse_interpolation):
            a = float(i) / mouse_interpolation
            x = int((1.0 - a) * mouse_pos[0] + a * previous_mouse_pos[0])
            y = int((1.0 - a) * mouse_pos[1] + a * previous_mouse_pos[1])
            on_mouse_click((x, y))
    else:
        on_mouse_click(mouse_pos)
    previous_mouse_pos = mouse_pos


def sparse_to_rgb(sparse_arr):
    t = np.repeat(sparse_arr, 3, axis=0)
    return np.transpose(t, (2, 1, 0))


def draw_sketch():
    pygame.surfarray.blit_array(input_window_surface, rgb_array)
    pygame.transform.scale(input_window_surface, (subwindow_width, subwindow_height), input_window)
    pygame.draw.rect(window, (0, 0, 0), (border_offset, border_offset, subwindow_width, subwindow_height), 1)


#Main loop
running = True
while running:
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

    if should_update:
        rgb_array = sparse_to_rgb(current_sketch)
        should_update = False
        
    window.fill(background_colour)
    draw_sketch()

    pygame.display.flip()
    pygame.time.wait(10)

