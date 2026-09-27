import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# Читает таблично заданную функцию из текстового файла.

# Формат файла:
#     первая значимая строка  — целое число N (количество узлов);
#     следующие N строк       — пары чисел 'x y', разделённые пробелом.

# Пустые строки и строки, начинающиеся с символа '#', игнорируются.
# Десятичный разделитель может быть как точкой, так и запятой.

# Возвращает два массива numpy: x_nodes и y_nodes
def read_data_from_file(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Файл '{filename}' не найден")

    # Читаем файл и оставляем только значимые строки
    with open(filename, 'r', encoding='utf-8') as f:
        lines = []
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            lines.append(line)

    if not lines:
        raise ValueError("Файл пуст или не содержит значимых строк")

    # Первая строка — количество узлов
    n = int(lines[0].split()[0])

    xs = []
    ys = []
    # Следующие n строк — координаты узлов
    for line in lines[1:n + 1]:
        parts = line.replace(',', '.').split()
        if len(parts) < 2:
            raise ValueError(f"Не хватает значений в строке: '{line}'")
        xs.append(float(parts[0]))
        ys.append(float(parts[1]))

    if len(xs) != n:
        raise ValueError(f"Ожидалось {n} узлов, а прочитано {len(xs)}")

    return np.array(xs, dtype=float), np.array(ys, dtype=float)

# Кусочно-линейная интерполяция
# Каждая пара соседних узлов соединяется отрезком прямой
def linear_interpolation(x_nodes, y_nodes, x):
    return np.interp(x, x_nodes, y_nodes)


# Интерполяционный многочлен Лагранжа

# Многочлен строится как линейная комбинация базисных многочленов l_i(x).
# Каждый l_i равен единице в своём узле x_i и нулю во всех остальных.
# Функция векторизована: x_arr — массив numpy, результат — тоже массив.
def lagrange_polynomial_vec(x_nodes, y_nodes, x_arr):
    n = len(x_nodes)
    result = np.zeros_like(x_arr, dtype=float)

    for i in range(n):
        # Начинаем с множителя y_i и домножаем на произведение дробей
        term = np.full_like(x_arr, y_nodes[i], dtype=float)
        for j in range(n):
            if i != j:
                term *= (x_arr - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term

    return result

# Вычисление таблицы разделённых разностей.

# Коэффициент coef[k] равен f[x0, x1, ..., xk] — разделённой разности
# порядка k, стоящей в верхней диагонали таблицы.
# Именно эти коэффициенты используются в многочлене Ньютона.
def divided_differences(x_nodes, y_nodes):
    n = len(x_nodes)
    coef = y_nodes.astype(float).copy()

    # Классическая схема вычисления разделённых разностей "снизу вверх"
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x_nodes[i] - x_nodes[i - j])

    return coef

# Интерполяционный многочлен Ньютона (формула 2.38 лекции).

# Используется общая форма через разделённые разности, поэтому
# функция работает и для неравномерной сетки.
# Для равномерной сетки результат совпадает с формулой (2.39).

# Многочлен вычисляется по схеме Горнера:
#     N(x) = c0 + (x - x0) * (c1 + (x - x1) * (c2 + ...))
# что экономит число операций и уменьшает погрешность округления.
def newton_polynomial_vec(x_nodes, coef, x_arr):
    n = len(x_nodes)
    result = np.full_like(x_arr, coef[0], dtype=float)
    product = np.ones_like(x_arr, dtype=float)

    for k in range(1, n):
        # Накапливаем произведение (x - x0)(x - x1)...(x - x_{k-1})
        product *= (x_arr - x_nodes[k - 1])
        result += coef[k] * product

    return result


def main():
    # Спрашиваем у пользователя имя файла с данными
    filename = input("Введите имя файла с данными (например, data.txt): ").strip()
    if not filename:
        filename = "data.txt"

    # Пытаемся прочитать данные; при ошибке аккуратно выходим
    try:
        x_nodes, y_nodes = read_data_from_file(filename)
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        sys.exit(1)

    # Выводим прочитанные узлы для контроля
    print(f"\nПрочитано {len(x_nodes)} узлов из файла '{filename}':")
    for xi, yi in zip(x_nodes, y_nodes):
        print(f"  x = {xi:8.4f}   y = {yi:8.4f}")

    # Границы интервала для построения графика
    x_min, x_max = x_nodes.min(), x_nodes.max()
    x_plot = np.linspace(x_min, x_max, 600)

    # Считаем значения всех трёх интерполянтов на плотной сетке
    y_linear = linear_interpolation(x_nodes, y_nodes, x_plot)
    y_lagrange = lagrange_polynomial_vec(x_nodes, y_nodes, x_plot)

    # Для многочлена Ньютона сначала находим коэффициенты,
    # а потом считаем значение многочлена
    coef_newton = divided_differences(x_nodes, y_nodes)
    y_newton = newton_polynomial_vec(x_nodes, coef_newton, x_plot)

    # Создаём одно окно с осями
    fig, ax = plt.subplots(figsize=(11, 7))

    # Узлы интерполяции показываем чёрными точками
    ax.plot(x_nodes, y_nodes, 'ko', markersize=9,
            label='Узлы интерполяции', zorder=5)

    # Линейная интерполяция — зелёный пунктир
    ax.plot(x_plot, y_linear, color='green', linestyle='--', linewidth=1.8,
            label='Линейная интерполяция')

    # Многочлен Лагранжа — красная сплошная линия
    ax.plot(x_plot, y_lagrange, color='red', linestyle='-', linewidth=2.2,
            label='Многочлен Лагранжа')

    # Многочлен Ньютона — синяя штрих-пунктирная линия.
    # Теоретически он совпадает с многочленом Лагранжа, поэтому
    # линии на графике практически сливаются.
    ax.plot(x_plot, y_newton, color='blue', linestyle='-.', linewidth=2.2,
            label='Многочлен Ньютона')

    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.set_title(f'Интерполяция функции (файл: {filename})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    plt.show()

    # Численное сравнение многочленов Лагранжа и Ньютона
    print("\nСравнение многочленов Лагранжа и Ньютона:")
    print(f"{'x':>8} {'Лагранж':>16} {'Ньютон':>16} {'|delta|':>12}")

    n_test = max(len(x_nodes) * 3, 15)
    for x_t in np.linspace(x_min, x_max, n_test):
        lag = lagrange_polynomial_vec(x_nodes, y_nodes, np.array([x_t]))[0]
        newt = newton_polynomial_vec(x_nodes, coef_newton, np.array([x_t]))[0]
        print(f"{x_t:8.4f} {lag:16.8f} {newt:16.8f} {abs(lag - newt):12.2e}")

    # Проверка, что в узлах интерполянты совпадают с табличными значениями
    print("\nЗначения в узлах интерполяции (должны совпасть с таблицей):")
    print(f"{'x':>8} {'y(таблица)':>16} {'Лагранж':>16} {'Ньютон':>16}")

    for i in range(len(x_nodes)):
        xi = x_nodes[i]
        lag = lagrange_polynomial_vec(x_nodes, y_nodes, np.array([xi]))[0]
        newt = newton_polynomial_vec(x_nodes, coef_newton, np.array([xi]))[0]
        print(f"{xi:8.4f} {y_nodes[i]:16.8f} {lag:16.8f} {newt:16.8f}")


if __name__ == "__main__":
    main()