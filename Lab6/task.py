import math
import numpy as np
import matplotlib.pyplot as plt

# Читает файл с исходными данными.
# Поддерживаются два формата:
#
#  1) Одно уравнение:
#     equation: выражение относительно x
#     segment: a b
#
#  2) Система из n уравнений с n неизвестными:
#     system:
#         выражение 1 (от x1, x2, ..., xn)
#         выражение 2
#         ...
#         выражение n
#     initial: x1_0 x2_0 ... xn_0
#
#  Возвращает словарь с ключами:
#     'type'      - 'single' или 'system'
#     'equation'  - строка с уравнением (для single)
#     'segment'   - кортеж (a, b) (для single)
#     'system'    - список из n строк (для system)
#     'initial'   - список из n чисел (для system)
#  Если файл не найден или данные неполные, возвращает None.
def read_input_file(filename):
    # Инициализируем словарь со значениями по умолчанию
    data = {
        'type': None,
        'equation': None,
        'segment': None,
        'system': None,
        'initial': None
    }

    # Пытаемся открыть файл
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
        return None

    # Переменные для отслеживания текущего режима чтения
    mode = None
    sys_lines = []

    # Построчно разбираем файл
    for line in lines:
        s = line.strip()
        # Пропускаем пустые строки и комментарии
        if not s or s.startswith('#'):
            continue

        low = s.lower()

        # Если встретили "equation:", значит это одно уравнение
        if low.startswith('equation:'):
            data['equation'] = s.split(':', 1)[1].strip()
            data['type'] = 'single'
            mode = None

        # Если "segment:", читаем отрезок
        elif low.startswith('segment:'):
            vals = s.split(':', 1)[1].replace(',', ' ').split()
            data['segment'] = (float(vals[0]), float(vals[1]))
            mode = None

        # Если "system:", начинаем читать систему уравнений
        elif low.startswith('system:'):
            mode = 'system'
            sys_lines = []
            data['type'] = 'system'

        # Если "initial:", читаем начальное приближение
        elif low.startswith('initial:'):
            vals = s.split(':', 1)[1].replace(',', ' ').split()
            data['initial'] = [float(v) for v in vals]
            mode = None

        # В режиме системы читаем строки с уравнениями
        elif mode == 'system':
            sys_lines.append(s)

    # Если были прочитаны строки системы, сохраняем их
    if sys_lines:
        data['system'] = sys_lines

    # Проверяем комплектность данных
    if data['type'] == 'single':
        if data['equation'] is None or data['segment'] is None:
            print("Ошибка: для одного уравнения нужны equation и segment.")
            return None
    elif data['type'] == 'system':
        if data['system'] is None or data['initial'] is None:
            print("Ошибка: для системы нужны уравнения и initial.")
            return None
        n = len(data['system'])
        # Проверяем, что число уравнений совпадает с числом начальных значений
        if len(data['initial']) != n:
            print(f"Ошибка: число уравнений ({n}) не совпадает с числом начальных значений ({len(data['initial'])}).")
            return None
    else:
        print("Ошибка: не указан тип (equation или system).")
        return None

    return data


# Создаёт функцию f(x) из строки с выражением.
# Поддерживаются элементарные математические функции из numpy.
def make_single_func(expr):
    # Создаём пространство имён с нужными математическими функциями
    ns = {
        name: getattr(np, name)
        for name in ['sin', 'cos', 'tan', 'arctan', 'arcsin', 'arccos',
                     'exp', 'log', 'sqrt', 'pi', 'e', 'abs']
    }
    # Запрещаем доступ к встроенным функциям Python для безопасности
    ns['__builtins__'] = None

    # Возвращаем лямбда-функцию от x
    def f(x):
        return eval(expr, ns, {'x': x})

    return f


# Создаёт список функций F_i(X) для системы из n уравнений.
# X - список (или массив) из n переменных.
# Переменные внутри выражений обозначаются как x1, x2, ..., xn.
def make_system_funcs(exprs):
    # Создаём пространство имён с математическими функциями
    ns = {
        name: getattr(np, name)
        for name in ['sin', 'cos', 'tan', 'arctan', 'arcsin', 'arccos',
                     'exp', 'log', 'sqrt', 'pi', 'e', 'abs']
    }
    ns['__builtins__'] = None

    # Фабрика функций: принимает строку выражения и возвращает функцию от X
    def make_func(expr):
        def F(X):
            # Формируем словарь локальных переменных x1, x2, ..., xn
            local = {f'x{i+1}': X[i] for i in range(len(X))}
            # Вычисляем выражение с этими переменными
            return eval(expr, ns, local)
        return F

    # Возвращаем список функций для каждого уравнения
    return [make_func(e) for e in exprs]


# Вспомогательная функция: рисует декартовы оси OX и OY через начало координат.
# Используется для придания графикам вида, привычного по учебникам.
def draw_axes(ax=None, color='black', linewidth=1.0):
    # Если ось не передана, берём текущую
    if ax is None:
        ax = plt.gca()

    # Скрываем стандартные рамки (spines)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Рисуем горизонтальную ось OX на уровне y = 0
    ax.axhline(0, color=color, linewidth=linewidth, zorder=1)
    # Рисуем вертикальную ось OY на уровне x = 0
    ax.axvline(0, color=color, linewidth=linewidth, zorder=1)

    # Убираем стандартные деления, оставляем только подписи
    ax.tick_params(axis='both', which='both', direction='inout', length=4)


# Метод бисекции (деления отрезка пополам).
# Находит корень уравнения f(x) = 0 на отрезке [a, b].
def bisection_method(f, a, b, eps=1e-6, max_iter=1000):
    # Вычисляем значения функции на концах отрезка
    fa = f(a)
    fb = f(b)

    # Проверяем, что на концах функция имеет разные знаки
    if fa * fb > 0:
        raise ValueError("На концах отрезка функция одного знака")

    # Начальное приближение - середина отрезка
    c = (a + b) / 2.0
    k = 0  # счётчик итераций

    # Основной цикл: пока отрезок длинный и значение функции не мало
    while abs(a - b) > eps and abs(f(c)) > eps and k < max_iter:
        fc = f(c)  # значение в середине

        # Если знак на [a, c] меняется, корень там, сдвигаем правую границу
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            # Иначе корень на [c, b], сдвигаем левую границу
            a = c
            fa = fc

        # Новая середина
        c = (a + b) / 2.0
        k += 1

    # Возвращаем количество итераций, приближённый корень и значение функции
    return k, c, f(c)


# Метод Брента (обратная параболическая интерполяция).
# Строит обратную параболу x = alpha + beta*y + gamma*y^2
# по трём точкам и берёт x = alpha как новое приближение.
def brent_method(f, a, b, eps=1e-6, max_iter=1000):
    # Начальные три точки: концы отрезка и середина
    x0, x1, x2 = a, b, (a + b) / 2.0
    y0, y1, y2 = f(x0), f(x1), f(x2)

    k = 0  # счётчик итераций

    # Основной цикл
    for k in range(1, max_iter + 1):
        # Строим матрицу A для системы уравнений относительно alpha, beta, gamma
        # Уравнения: x_i = alpha + beta*y_i + gamma*y_i^2
        A = np.array([
            [1.0, y0, y0 * y0],
            [1.0, y1, y1 * y1],
            [1.0, y2, y2 * y2]
        ], dtype=float)

        # Вектор правой части: известные x_i
        B = np.array([x0, x1, x2], dtype=float)

        # Решаем систему для коэффициентов параболы
        try:
            alpha, beta, gamma = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            # Если матрица вырождена, берём середину между x1 и x2
            x3 = (x1 + x2) / 2.0
        else:
            # Новое приближение - значение параболы при y=0
            x3 = alpha

        # Сдвигаем точки: выбрасываем самую старую, добавляем новую
        x0, y0 = x1, y1
        x1, y1 = x2, y2
        x2 = x3
        y2 = f(x2)

        # Проверяем сходимость по относительному изменению x
        if abs(x2 - x1) / max(abs(x2), 1e-15) < eps:
            break

    # Возвращаем количество итераций, корень и значение функции
    return k, x2, f(x2)


# Сканирует отрезок [a, b] с шагом step, ищет смену знака функции.
# На каждом интервале, где знак меняется, уточняет корень выбранным методом.
def find_all_roots(f, a, b, method, eps=1e-6, step=0.05):
    roots = []          # список найденных корней
    x = a               # текущая точка сканирования
    prev_f = f(a)       # значение функции в текущей точке

    # Если на левом конце уже корень, добавляем его
    if abs(prev_f) < eps:
        roots.append(a)

    # Идём по отрезку с шагом step
    while x < b:
        x_next = min(x + step, b)   # следующая точка (не выходим за b)
        f_next = f(x_next)          # значение функции в ней

        # Если значение близко к нулю, считаем это корнем
        if abs(f_next) < eps:
            # Добавляем, если такого корня ещё нет
            if all(abs(x_next - rr) > 1e-6 for rr in roots):
                roots.append(x_next)

        # Если знак сменился, уточняем корень выбранным методом
        elif prev_f * f_next < 0:
            try:
                if method == 'brent':
                    _, r, _ = brent_method(f, x, x_next, eps)
                else:
                    _, r, _ = bisection_method(f, x, x_next, eps)

                # Добавляем, если корень новый
                if all(abs(r - rr) > 1e-6 for rr in roots):
                    roots.append(r)

            except Exception as e:
                # Если метод не сошёлся, выводим предупреждение
                print(f"  Предупреждение на [{x:.4f}, {x_next:.4f}]: {e}")

        # Переходим к следующему интервалу
        x = x_next
        prev_f = f_next

    # Проверяем правый конец отрезка
    if abs(f(b)) < eps and all(abs(b - rr) > 1e-6 for rr in roots):
        roots.append(b)

    # Возвращаем отсортированный список корней
    return sorted(roots)

# Решает одно уравнение выбранным методом.
# Если separate == True, то сначала выполняется отделение корней
# с шагом h, затем на каждом подынтервале применяется метод.
# Если separate == False, метод применяется напрямую на [a, b].
def solve_single_equation(f, a, b, method, eps, separate, h=0.05):
    if separate:
        # Отделяем корни сканированием с шагом h
        roots = find_all_roots(f, a, b, method, eps, step=h)

        print("\nНайденные корни (с отделением):")
        for r in roots:
            print(f"  x = {r:.10f}, f(x) = {f(r):.3e}")

        plot_function(f, a, b, roots,
                      f'Метод {"Брента" if method == "brent" else "бисекции"} с отделением')
    else:
        # Применяем метод напрямую на [a, b]
        try:
            if method == 'brent':
                k, r, fr = brent_method(f, a, b, eps)
            else:
                k, r, fr = bisection_method(f, a, b, eps)
        except ValueError as e:
            print(f"Ошибка: {e}")
            return

        print(f"\nКорень (без отделения): x = {r:.10f}, f(x) = {fr:.3e}")
        print(f"Количество итераций: {k}")

        plot_function(f, a, b, [r],
                      f'Метод {"Брента" if method == "brent" else "бисекции"} без отделения')


# Решает систему из n нелинейных уравнений методом Ньютона.
# Итерационная формула:
#        J(X) * dX = -F(X)
#   где J - матрица Якоби (n x n), F - вектор невязок (n).
# Новое приближение: X := X + dX.
def solve_system_newton(data):
    exprs = data['system']              # список строк с уравнениями
    X = np.array(data['initial'], dtype=float)  # начальное приближение
    n = len(X)                          # размерность системы

    # Запрашиваем точность и максимальное число итераций
    eps = float(input("Введите eps (например, 1e-6): ") or "1e-6")
    max_iter = int(input("Введите максимальное число итераций (например, 50): ") or "50")

    # Создаём функции F_i(X)
    funcs = make_system_funcs(exprs)

    print(f"\nРазмерность системы: n = {n}")
    print("Итерации метода Ньютона:")

    # Основной цикл метода Ньютона
    for k in range(1, max_iter + 1):
        # Вычисляем вектор невязок F(X)
        F = np.array([f(X) for f in funcs], dtype=float)

        # Численное построение матрицы Якоби
        # Используем центральную разность для каждой частной производной
        h = 1e-6
        J = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[j] += h
                X_minus[j] -= h
                J[i, j] = (funcs[i](X_plus) - funcs[i](X_minus)) / (2 * h)

        # Решаем линейную систему J * dX = -F
        try:
            dX = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            print("Матрица Якоби вырождена. Итерации остановлены.")
            break

        # Обновляем приближение
        X = X + dX

        # Контроль сходимости
        max_dx = float(np.max(np.abs(dX)))
        max_F = float(np.max(np.abs(F)))

        # Выводим информацию о текущей итерации
        print(f"  k={k:2d}: " +
              ", ".join(f"x{i+1}={X[i]:.8f}" for i in range(n)) +
              f", max|dx|={max_dx:.3e}, max|F|={max_F:.3e}")

        # Если поправки малы, завершаем
        if max_dx < eps:
            print("Достигнута заданная точность.")
            break

    # Вывод решения
    print("\nРешение системы:")
    for i in range(n):
        print(f"  x{i+1} = {X[i]:.10f}")

    # Вывод невязок
    F_final = np.array([f(X) for f in funcs], dtype=float)
    print("Невязки:")
    for i in range(n):
        print(f"  F{i+1} = {F_final[i]:.3e}")

    # График строим только для системы из двух уравнений
    if n == 2:
        F1, F2 = funcs

        # Создаём сетку для построения линий уровня
        xs = np.linspace(-1.5, 1.5, 400)
        ys = np.linspace(0, 1.5, 400)
        Xg, Yg = np.meshgrid(xs, ys)

        # Вычисляем значения F1 и F2 на сетке
        Z1 = np.zeros_like(Xg)
        Z2 = np.zeros_like(Xg)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                Z1[i, j] = F1([Xg[i, j], Yg[i, j]])
                Z2[i, j] = F2([Xg[i, j], Yg[i, j]])

        # Создаём фигуру и оси
        fig, ax = plt.subplots()

        # Рисуем контуры F1=0 (красный) и F2=0 (синий)
        ax.contour(Xg, Yg, Z1, levels=[0], colors='r')
        ax.contour(Xg, Yg, Z2, levels=[0], colors='b')

        # Отмечаем найденное решение зелёной точкой
        ax.plot(X[0], X[1], 'go', markersize=8, label='решение')

        # Рисуем декартовы оси OX и OY через начало координат
        draw_axes(ax)

        # Подписываем оси и добавляем заголовок
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Система: красный F1=0, синий F2=0')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        plt.show()


# Строит график функции f(x) на отрезке [a, b] и отмечает найденные корни
def plot_function(f, a, b, roots, title):
    # Генерируем точки по оси X
    xs = np.linspace(a, b, 500)
    # Вычисляем значения функции
    ys = [f(x) for x in xs]

    # Создаём фигуру и оси
    fig, ax = plt.subplots()

    # Рисуем график функции
    ax.plot(xs, ys, label='f(x)', color='tab:blue')

    # Рисуем декартовы оси OX и OY через начало координат
    draw_axes(ax)

    # Отмечаем каждый корень красной точкой и подписываем значение
    for r in roots:
        ax.plot(r, f(r), 'ro', zorder=5)
        ax.annotate(
            f'{r:.4f}',
            (r, f(r)),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            color='darkred'
        )

    # Подписываем оси и добавляем заголовок
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.show()


def main():
    # Запрашиваем имя файла с данными
    filename = input("Введите имя файла с исходными данными (например, input.txt): ").strip()

    # Читаем данные
    data = read_input_file(filename)
    if data is None:
        return

    # Если это одно уравнение
    if data['type'] == 'single':
        expr = data['equation']
        a, b = data['segment']
        f = make_single_func(expr)

        print(f"\nУравнение: {expr} = 0")
        print(f"Отрезок: [{a}, {b}]")

        # Меню для одного уравнения
        while True:
            print("\nМеню:")
            print("1 - Отделять корни на отрезке")
            print("2 - Решать без отделения корней (на заданном отрезке)")
            print("0 - Выход")

            choice = input("Выберите пункт: ").strip()

            if choice in ('1', '2'):
                separate = (choice == '1')

                # Шаг отделения спрашиваем только если выбрано отделение
                h = 0.05
                if separate:
                    h_input = input("Введите шаг отделения h (например, 0.05): ").strip()
                    h = float(h_input) if h_input else 0.05

                # Выбор метода решения
                print("\nВыберите метод:")
                print("1 - Метод Брента")
                print("2 - Метод бисекции")
                m = input("Ваш выбор: ").strip()

                if m == '1':
                    method = 'brent'
                elif m == '2':
                    method = 'bisection'
                else:
                    print("Неверный метод.")
                    continue

                eps = float(input("Введите eps (например, 1e-6): ") or "1e-6")

                solve_single_equation(f, a, b, method, eps, separate, h)

            elif choice == '0':
                print("Выход...")
                break

            else:
                print("Неверный пункт меню.")

    # Если это система уравнений
    elif data['type'] == 'system':
        solve_system_newton(data)


if __name__ == "__main__":
    main()