import numpy as np
import matplotlib.pyplot as plt

# Funkcja do interpolacji najbliższego sąsiada
def nearest_neighbor_interpolation(x, y, x_new):
    y_new = []  # Tworzenie pustej listy na nowe wartości y
    for xi in x_new:  # Przechodzenie przez nowe wartości x
        closest_idx = np.argmin(np.abs(x - xi))  # Znalezienie indeksu najbliższego sąsiada
        y_new.append(y[closest_idx])  # Dodanie wartości y dla najbliższego sąsiada
    return np.array(y_new)  # Zwrócenie nowych wartości y jako tablicy

# Funkcja do interpolacji liniowej
def linear_interpolation(x, y, x_new):
    y_new = []  # Tworzenie pustej listy na nowe wartości y
    for xi in x_new:  # Przechodzenie przez nowe wartości x
        if xi <= x[0]:  # Sprawdzenie, czy wartość x_new jest mniejsza niż pierwsza wartość w x
            y_new.append(y[0])  # Jeśli tak, dodanie pierwszej wartości y
        elif xi >= x[-1]:  # Sprawdzenie, czy wartość x_new jest większa niż ostatnia wartość w x
            y_new.append(y[-1])  # Jeśli tak, dodanie ostatniej wartości y
        else:
            idx = np.where(x > xi)[0][0]  # Znalezienie indeksu, gdzie wartość x przekracza x_new
            x_left, x_right = x[idx - 1], x[idx]  # Określenie sąsiednich wartości x
            y_left, y_right = y[idx - 1], y[idx]  # Określenie sąsiednich wartości y
            slope = (y_right - y_left) / (x_right - x_left)  # Obliczenie nachylenia prostej
            y_interp = y_left + slope * (xi - x_left)  # Obliczenie interpolowanej wartości y
            y_new.append(y_interp)  # Dodanie interpolowanej wartości y
    return np.array(y_new)  # Zwrócenie nowych wartości y jako tablicy

# Dane wejściowe
x1 = np.linspace(0, 5, 10)  # Przykładowe wartości x
y1 = np.sin(x1)  # Wartości y dla funkcji sinus
x2 = np.linspace(0, 5, 100)  # Nowe wartości x dla interpolacji

# Tworzenie wykresu dla (x1, y1)
plt.plot(x1, y1, '.-r')
plt.xlabel('x1')
plt.ylabel('y1')
plt.title('Wykres x1 na y1')
plt.savefig("Wykres oryginalny")
plt.show(block=False)

# Interpolacja najbliższego sąsiada
y2_nearest = nearest_neighbor_interpolation(x1, y1, x2)  # Wywołanie funkcji interpolacji najbliższego sąsiada

# Tworzenie nowego wykresu dla (x2, y2) - Interpolacja najbliższego sąsiada
plt.figure()
plt.plot(x2, y2_nearest, '.-b')
plt.xlabel('x2')
plt.ylabel('y2')
plt.title('Interpolacja najbliższego sąsiada - Wykres x2 na y2')
plt.savefig("Wykres najblizszego sasiada")
plt.show(block=False)

# Interpolacja liniowa
y2_linear = linear_interpolation(x1, y1, x2)  # Wywołanie funkcji interpolacji liniowej

# Tworzenie nowego wykresu dla (x2, y2) - Interpolacja liniowa
plt.figure()
plt.plot(x2, y2_linear, '.-g')
plt.xlabel('x2')
plt.ylabel('y2')
plt.title('Interpolacja liniowa - Wykres x2 na y2')
plt.savefig("Wykres interpolacji liniowej")
plt.show()
