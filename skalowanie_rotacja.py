import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

# Wczytanie obrazu
img = cv2.cvtColor(cv2.imread('demo_2.jpg'), cv2.COLOR_BGR2RGB)
original_image = img.copy()  # Utworzenie kopii oryginalnego obrazu
heightO, widthO = original_image.shape[:2]

# Funkcja do zmniejszania obrazu o połowę
def resize_down(image):
    height, width = image.shape[:2]
    return image[::2, ::2]  # Zmniejszenie obrazu o połowę w każdym wymiarze

# Funkcja do powiększania obrazu o dwukrotność
def resize_up(image):
    height, width = image.shape[:2]
    new_height = height * 3
    new_width = width * 3
    result = np.zeros((new_height, new_width, 3), dtype=np.uint8)  # Nowa macierz na powiększony obraz
    # Iteracja po nowym obrazie i przypisanie wartości pikseli na podstawie obrazu zmniejszonego
    for i in range(new_height):
        for j in range(new_width):
            result[i, j] = image[i // 3, j // 3]  # Przypisanie piksela z obrazu zmniejszonego
    return result


# Funkcja do obrotu obrazu z zastosowaniem interpolacji
def rotate_image_with_interpolation(image, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    cosine_angle = math.cos(angle_radians)
    sine_angle = math.sin(angle_radians)
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1  # Liczba kanałów

    new_height = round(abs(image.shape[0] * cosine_angle) + abs(image.shape[1] * sine_angle)) + 1
    new_width = round(abs(image.shape[1] * cosine_angle) + abs(image.shape[0] * sine_angle)) + 1

    rotated_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    original_center_height = (image.shape[0] - 1) / 2
    original_center_width = (image.shape[1] - 1) / 2
    new_center_height = (new_height - 1) / 2
    new_center_width = (new_width - 1) / 2

    for w in range(new_height):
        for k in range(new_width):
            # Obliczenie współrzędnych piksela obrazu po rotacji
            x_rotated = (k - new_center_width) * cosine_angle + (w - new_center_height) * sine_angle
            y_rotated = (w - new_center_height) * cosine_angle - (k - new_center_width) * sine_angle

            # Dostosowanie do współrzędnych obrazu przed rotacją
            x_rotated += original_center_width
            y_rotated += original_center_height

            # Sprawdzenie, czy piksel mieści się w oryginalnych wymiarach obrazu
            if 0 <= x_rotated < width - 1 and 0 <= y_rotated < height - 1:
                x1, y1 = int(x_rotated), int(y_rotated)
                x2, y2 = x1 + 1, y1 + 1

                # Obliczenie wag dla interpolacji dwuliniowej
                alpha = x_rotated - x1
                beta = y_rotated - y1

                # Interpolacja dwuliniowa dla każdego kanału
                for c in range(channels):
                    rotated_image[w, k, c] = (1 - alpha) * (1 - beta) * image[y1, x1, c] + alpha * (1 - beta) * image[y1, x2, c] \
                                      + (1 - alpha) * beta * image[y2, x1, c] + alpha * beta * image[y2, x2, c]

    # Interpolacja obrazu po rotacji, aby przywrócić go do oryginalnych wymiarów
    restored_resized_image = interpolate_up(rotated_image, image.shape)
    return restored_resized_image


# Funkcja do interpolacji przywracającej obraz do oryginalnego rozmiaru
def interpolate_up(image, original_shape):
    new_height, new_width = image.shape[:2]
    original_height, original_width = original_shape[:2]

    result = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Obliczenie współczynników do interpolacji
    x_ratio = original_width / new_width
    y_ratio = original_height / new_height

    for i in range(original_height):
        for j in range(original_width):
            # Wyznaczenie odpowiadającego piksela z obrazu zmniejszonego
            x = int(j / x_ratio)
            y = int(i / y_ratio)

            # Przypisanie wartości piksela z obrazu zmniejszonego do odpowiadającego mu piksela w obrazie powiększonym
            result[i, j] = image[y, x]

    return result


# Użycie funkcji do zmniejszenia i powiększenia obrazu
smaller_image = resize_down(img.copy())
larger_image = resize_up(smaller_image.copy())

# # Zmiana rozmiaru obrazu na 512 x 512 pikseli
# resized_image = cv2.resize(img, (512, 512))

# Przywrócenie obrazu do oryginalnego rozmiaru po interpolacji
resized_image = interpolate_up(larger_image, img.shape)

# Obrót obrazu o 30 stopni
rotated_image = rotate_image_with_interpolation(original_image, 30)


# Oryginalny obraz
plt.imshow(original_image)
plt.title('Oryginalny')
plt.show()

# Pomniejszony obraz
plt.imshow(smaller_image)
plt.title('Pomniejszony')
plt.show()

# Powiększony obraz
plt.imshow(larger_image)
plt.title('Powiększony')
plt.show()

# Obraz przeskalowany
plt.imshow(resized_image)
plt.title('Przeskalowany')
plt.show()

# Obrócony o 30 stopni obraz
plt.imshow(rotated_image)
plt.title('Obrócony o 30 stopni')
plt.show()
