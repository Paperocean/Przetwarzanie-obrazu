import cv2
import numpy as np
import matplotlib.pyplot as plt

def closest_neighBayer(IMG, BAYERMASK):
    height, width, _ = IMG.shape
    demosaiced_image = np.zeros_like(IMG, dtype=np.uint8)

    for w in range(height):
        for s in range(width):
            for k in range(3):  # kolory
                if BAYERMASK[k, w % 2, s % 2] == 1:
                    demosaiced_image[w, s, k] = IMG[w, s, k]
                else:
                    closest_w = w + 1 if w % 2 == 0 else w - 1
                    closest_s = s + 1 if s % 2 == 0 else s - 1
                    if 0 <= closest_w < height and 0 <= closest_s < width:
                        demosaiced_image[w, s, k] = IMG[closest_w, closest_s, k]
                    else:
                        # obsługa sytuacji, gdy indeksy są poza zakresem
                        demosaiced_image[w, s, 0] = demosaiced_image[w, s, 1] = demosaiced_image[w, s, 2] = 255

    return demosaiced_image


def closest_neighXTrans(XIMG, XTRANSMASK):
    height, width, _ = XIMG.shape
    demosaiced_image = np.zeros_like(XIMG, dtype=np.uint8)

    for w in range(height):
        for s in range(width):
            for k in range(3):  # kolory
                if XTRANSMASK[k, w % 6, s % 6] == 1:
                    demosaiced_image[w, s, k] = XIMG[w, s, k]
                else:
                    closest_w = w + 1 if w % 2 == 0 else w - 1
                    closest_s = s + 1 if s % 2 == 0 else s - 1
                    if 0 <= closest_w < height and 0 <= closest_s < width:
                        demosaiced_image[w, s, k] = XIMG[closest_w, closest_s, k]
                    else:
                        # obsługa sytuacji, gdy indeksy są poza zakresem
                        demosaiced_image[w, s, 0] = demosaiced_image[w, s, 1] = demosaiced_image[w, s, 2] = 255

    return demosaiced_image


# Wczytanie obrazu i konwersja na przestrzeń kolorów RGB
im = cv2.cvtColor(cv2.imread('demo_2.jpg'), cv2.COLOR_BGR2RGB)

# Przykładowa maska Bayera
BayerMask = np.array([[[0, 1], [0, 0]],
                      [[1, 0], [0, 1]],
                      [[0, 0], [1, 0]]], np.uint8)

# Użycie maski Bayera na obrazie
filtered_image_bayer = np.zeros_like(im)
height, width = im.shape[:2]

for h in range(height):
    for w in range(width):
        for c in range(3):  # Dla każdego kanału koloru
            filtered_image_bayer[h, w, c] = BayerMask[c, h % 2, w % 2] * im[h, w, c]

# Demozaikowanie obrazu Bayera
demosaiced_bayer = closest_neighBayer(im, BayerMask)

# Przykładowa maska X-Trans
XTransMask = np.array([[[0, 0, 0, 0, 1, 0],
                        [1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1],
                        [0, 1, 0, 0, 0, 0]],            #R
                       [[1, 0, 1, 1, 0, 1],
                        [0, 1, 0, 0, 1, 0],
                        [1, 0, 1, 1, 0, 1],
                        [1, 0, 1, 1, 0, 1],
                        [0, 1, 0, 0, 1, 0],
                        [1, 0, 1, 1, 0, 1]],            #G
                       [[0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0]]], np.uint8) #B

# Użycie maski X-Trans na obrazie
filtered_image_xtrans = np.zeros_like(im)

height, width = im.shape[:2]

for h in range(height):
    for w in range(width):
        for c in range(3):  # Dla każdego kanału koloru
            filtered_image_xtrans[h, w, c] = XTransMask[c, h % 6, w % 6] * im[h, w, c]

# Demozaikowanie X-Trans
demosaiced_xtrans = closest_neighXTrans(im, XTransMask)

# Wyświetlenie oryginalnego i przefiltrowanego obrazu
plt.figure(figsize=(15, 6))

plt.subplot(1, 5, 1)
plt.title('Oryginalny obraz')
plt.imshow(im)
plt.axis('off')

plt.subplot(1, 5, 2)
plt.title('Przefiltrowany obraz z maską Bayera')
plt.imshow(filtered_image_bayer)
plt.axis('off')

plt.subplot(1, 5, 3)
plt.title('Przefiltrowany obraz z maską X-Trans')
plt.imshow(filtered_image_xtrans)
plt.axis('off')

plt.subplot(1, 5, 4)
plt.title('Demozaikowanie obrazu Bayera')
plt.imshow(demosaiced_bayer)
plt.axis('off')

plt.subplot(1, 5, 5)
plt.title('Demozaikowanie obrazu X-Trans')
plt.imshow(demosaiced_xtrans)
plt.axis('off')

plt.tight_layout()
plt.show()