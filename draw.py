
""" Draw a digit within the canvas and ask the trained model to predict a value """

import pygame
import numpy as np
from math import sqrt

import test
from ANN.ann import *


WIDTH = 800
HEIGHT = 600

pygame.init()
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "Canvas [%i X %i],  draw a digit!" % (WIDTH, HEIGHT))

window_matrix = []
image_scaled = False
run = False


def draw(position):
    def distance(x1, y1, x2, y2):
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def check_cors(x, y):
        return WIDTH > x >= 0 and HEIGHT > y >= 0

    global window_matrix
    x_cor = position[0]
    y_cor = position[1]

    brush_size = 54
    for i in range(x_cor - brush_size // 2, x_cor + brush_size // 2):
        for j in range(y_cor - brush_size // 2, y_cor + brush_size // 2):
            if check_cors(i, j):
                dist = distance(x_cor, y_cor, i, j)
                if dist <= brush_size // 2:
                    pixel_rgb = (255, 255, 255)
                    pygame.draw.rect(window, pixel_rgb, (i, j, 1, 1))
                    window_matrix[j][i] = 255

    pygame.display.update()


def display_scaled_image(scaled_image_matrix, predicted_digit=-1):
    scaled_image_window_width = 700
    scaled_image_window_height = 720
    offset_x = 70
    offset_y = 40

    scaled_image_window = pygame.display.set_mode(
        (scaled_image_window_width, scaled_image_window_height))

    for i in range(0, 28):
        for j in range(0, 28):
            pixel_rgb = (
                scaled_image_matrix[i][j], scaled_image_matrix[i][j], scaled_image_matrix[i][j])
            pixel_size = 20
            position_x = offset_x + j * pixel_size
            position_y = offset_y + i * pixel_size
            pygame.draw.rect(scaled_image_window, pixel_rgb,
                             (position_x, position_y, pixel_size, pixel_size))

    pygame.display.set_caption("Canvas input scaled down to [28 X 28]")

    message = "Model unable to identify the digit, Please Retry!" if predicted_digit == - \
        1 else "Predicted digit  =  %i." % predicted_digit
    font = pygame.font.SysFont('rasa', 32)
    text = font.render(message, True, (180, 180, 180), (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = (350, 650)
    window.blit(text, text_rect)

    pygame.display.update()


def scale_down_image(image_matrix):
    scaled_image_matrix = []
    for i in range(0, 28):
        scaled_image_matrix.append([])
        for j in range(0, 28):
            scaled_image_matrix[i].append(0)

    fill_x_start, fill_y_start = 8, 8
    scale_factor = len(image_matrix[0]) / 20

    if scale_factor == int(scale_factor):
        scale_factor = int(scale_factor)
        for i in range(0, 20):
            for j in range(0, 20):
                greyscale_sum = 0
                pixel_count = 0
                for p in range(i * scale_factor, i * scale_factor + scale_factor):
                    for q in range(j * scale_factor, j * scale_factor + scale_factor):
                        greyscale_sum += image_matrix[p][q]
                        pixel_count += 1

                scaled_image_matrix[fill_y_start // 2 + i][fill_x_start //
                                                           2 + j] = int(greyscale_sum / pixel_count)

    else:
        for i in range(0, 20):
            for j in range(0, 20):
                greyscale_sum = 0
                pixel_count = 0
                for p in range(int(i * scale_factor), int(i * scale_factor) + int(scale_factor + 1)):
                    for q in range(int(j * scale_factor), int(j * scale_factor) + int(scale_factor + 1)):
                        greyscale_sum += image_matrix[p][q]
                        pixel_count += 1

                scaled_image_matrix[fill_y_start // 2 + i][fill_x_start //
                                                           2 + j] = int(greyscale_sum / pixel_count)

    return scaled_image_matrix


def is_canvas_empty():
    global window_matrix
    isempty = True
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if window_matrix[i][j] != 0:
                isempty = False
                break

    return isempty


def check_key_pressed():
    global window_matrix, run, image_scaled
    keys = pygame.key.get_pressed()

    if keys[pygame.K_RETURN]:
        if is_canvas_empty():
            print("Please draw something first!")
        else:
            (vertical_start, vertical_end, horizontal_start,
             horizontal_end) = (HEIGHT, -1, WIDTH, -1)
            for i in range(0, HEIGHT):
                for j in range(0, WIDTH):
                    if window_matrix[i][j] != 0:
                        vertical_start = min(i, vertical_start)
                        vertical_end = max(i, vertical_end)
                        horizontal_start = min(j, horizontal_start)
                        horizontal_end = max(j, horizontal_end)

            horizontal_dim = horizontal_end - horizontal_start + 1
            vertical_dim = vertical_end - vertical_start + 1
            image_dim = max(horizontal_dim, vertical_dim)

            image_matrix = []
            for i in range(0, image_dim):
                image_matrix.append([])
                for j in range(0, image_dim):
                    image_matrix[i].append(0)

            if horizontal_dim < image_dim:
                diff = image_dim - horizontal_dim
                for i in range(vertical_start, vertical_end + 1):
                    for j in range(horizontal_start, horizontal_end + 1):
                        image_matrix[i - vertical_start][diff // 2 +
                                                         j - horizontal_start] = window_matrix[i][j]

            else:
                diff = image_dim - vertical_dim
                for i in range(vertical_start, vertical_end + 1):
                    for j in range(horizontal_start, horizontal_end + 1):
                        image_matrix[diff // 2 + i - vertical_start][j -
                                                                     horizontal_start] = window_matrix[i][j]

            scaled_down_image = scale_down_image(image_matrix)
            window_matrix = scaled_down_image
            display_scaled_image(
                scaled_down_image, test.get_digit(scaled_down_image))
            image_scaled = True


def init_window_matrix():
    global window_matrix

    for i in range(0, HEIGHT):
        window_matrix.append([])
        for j in range(0, WIDTH):
            window_matrix[i].append(0)
    return window_matrix


def main():
    global window_matrix, run

    window_matrix = init_window_matrix()
    run = True
    pygame.display.update()
    click_triggers = 0

    while run:
        for event in pygame.event.get():
            if not image_scaled:
                check_key_pressed()
            if pygame.mouse.get_pressed()[0]:
                try:
                    event.pos
                except AttributeError:
                    print("Mouse pointer outside window")
                else:
                    if not image_scaled and click_triggers == 0:
                        draw(event.pos)
                    click_triggers += 1
                    click_triggers %= 2
            if event.type == pygame.QUIT:
                run = False


if __name__ == '__main__':
    main()






