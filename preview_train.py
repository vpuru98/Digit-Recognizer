
"""
Preview first hundred images of the training set
Use forward and backward arrow keys to preview next and previous images

"""

import pygame
from utility import get_training_image_vector, get_training_image_label

WIDTH = 700
HEIGHT = 700

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Preview Training Set : ")

display_image_number = 1
display_image_vector = []
for i in range(0, 784):
    display_image_vector.append(0)


def retrieve_image(image_number):
    global display_image_vector
    display_image_vector = get_training_image_vector(image_number)
    image_label = get_training_image_label(image_number)
    pygame.display.set_caption(
        "Preview Training Set : Displaying image [%i] with label %i" % (image_number, image_label))
    redraw_screen()


def redraw_screen():
    for index in range(0, 784):
        pixel_row = index // 28
        pixel_column = index % 28
        pixel_color = (
            display_image_vector[index], display_image_vector[index], display_image_vector[index])
        pixel_width = 20
        pixel_height = 20
        position_x = 70 + pixel_column * pixel_width
        position_y = 70 + pixel_row * pixel_height
        pygame.draw.rect(window, pixel_color, (position_x,
                                               position_y, pixel_width, pixel_height))
        pygame.display.update()


def key_presses():
    global display_image_number
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        display_image_number = ((display_image_number - 1) - 1 + 100) % 100 + 1
        retrieve_image(display_image_number)
    if keys[pygame.K_RIGHT]:
        display_image_number = ((display_image_number - 1) + 1 + 100) % 100 + 1
        retrieve_image(display_image_number)


def main():
    run = True
    retrieve_image(display_image_number)
    while run:
        key_presses()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False


if __name__ == '__main__':
    main()




