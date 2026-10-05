import math
import numpy as np
import matplotlib.pyplot as plt

# пропорция золотого сечения
PHI = (1 + math.sqrt(5)) / 2


def read_input_file(filename):
    # читает файл и возвращает словарь с данными
    data = {
        'function': None,  # строка с выражением одной переменной
        'segment': None,  # отрезок [a, b] для одной переменной
        'multifunction': None,  # строка с выражением многих переменных
        'initial': None,  # начальная точка для многомерной задачи
        'eps': 1e-6,  # точность по умолчанию
    }

    try:
        # открываем файл на чтение
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"файл '{filename}' не найден")
        return None

    # построчный разбор файла
    for line in lines:
        s = line.strip()  # убираем пробелы по краям
        if not s or s.startswith('#'):  # пустые строки и комментарии пропускаем
            continue
        low = s.lower()  # регистронезависимое сравнение

        if low.startswith('multifunction:'):  # многомерная функция
            data['multifunction'] = s.split(':', 1)[1].strip()
        elif low.startswith('function:'):  # одномерная функция
            data['function'] = s.split(':', 1)[1].strip()
        elif low.startswith('segment:'):  # отрезок [a, b]
            vals = s.split(':', 1)[1].replace(',', ' ').split()
            data['segment'] = (float(vals[0]), float(vals[1]))
        elif low.startswith('initial:'):  # начальная точка X0
            vals = s.split(':', 1)[1].replace(',', ' ').split()
            data['initial'] = [float(v) for v in vals]
        elif low.startswith('eps:'):  # точность eps
            data['eps'] = float(s.split(':', 1)[1].strip())

    # проверка, что задана хоть какая-то функция
    if data['function'] is None and data['multifunction'] is None:
        print("в файле не найдена функция")
        return None
    return data


def make_single_func(expr):
    # собирает f(x) из строки-выражения через eval
    ns = {
        name: getattr(np, name)
        for name in ['sin', 'cos', 'tan', 'arctan', 'arcsin', 'arccos',
                     'exp', 'log', 'sqrt', 'pi', 'e', 'abs']
    }
    ns['__builtins__'] = None

    def f(x):
        return eval(expr, ns, {'x': x})

    return f


def make_multi_func(expr):
    # собирает f(x1, x2, ..., xn) из строки-выражения
    ns = {
        name: getattr(np, name)
        for name in ['sin', 'cos', 'tan', 'arctan', 'arcsin', 'arccos',
                     'exp', 'log', 'sqrt', 'pi', 'e', 'abs']
    }
    ns['__builtins__'] = None

    def f(X):
        # формируем словарь локальных переменных x1, x2, ..., xn
        local = {f'x{i+1}': X[i] for i in range(len(X))}
        return eval(expr, ns, local)

    return f


def draw_axes(ax=None, color='black', linewidth=1.0):
    # рисует оси OX и OY через ноль
    if ax is None:
        ax = plt.gca()
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axhline(0, color=color, linewidth=linewidth, zorder=1)
    ax.axvline(0, color=color, linewidth=linewidth, zorder=1)
    ax.tick_params(axis='both', which='both', direction='inout', length=4)


def detect_extremum_type(f, x):
    # определяет тип экстремума по второй производной (численно)
    h = 1e-4
    second = (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    return 'min' if second > 0 else 'max'


def golden_section(f, a, b, eps=1e-6, max_iter=10000):
    # метод золотого сечения
    # шаг 1: заданы начальные границы [a, b] и точность eps
    # шаг 2: рассчитываем начальные точки деления
    # x1 = b - (b - a)/phi, x2 = a + (b - a)/phi
    x1 = b - (b - a) / PHI
    x2 = a + (b - a) / PHI
    # значения целевой функции в этих точках
    y1 = f(x1)
    y2 = f(x2)
    k = 1  # первая итерация уже сделана
    # шаг 3: проверяем условие остановки и цикл
    while abs(a - b) >= eps and k < max_iter:
        # шаг 2: если y1 ≥ y2 — сдвигаем левую границу
        if y1 >= y2:
            a = x1
            # используем свойство золотого сечения: x1_новое = x2_старое
            x1 = x2
            y1 = y2
            x2 = a + (b - a) / PHI  # пересчитываем только x2
            y2 = f(x2)
        else:
            # иначе сдвигаем правую границу
            b = x2
            x2 = x1
            y2 = y1
            x1 = b - (b - a) / PHI  # пересчитываем только x1
            y1 = f(x1)
        k += 1
    # шаг 3: x = (a + b)/2 и остановка
    x = (a + b) / 2.0
    return x, f(x), k


def brent(f, a, b, eps=1e-6, max_iter=1000):
    # метод Брента / параболической интерполяции
    # парабола y = f(x) = alpha + beta*x + gamma*x**2, строится по трём точкам
    # нулевая итерация: x0 = a, x1 = b, x2 = (a + b)/2
    x0, x1, x2 = a, b, (a + b) / 2.0
    y0, y1, y2 = f(x0), f(x1), f(x2)
    k = 0
    # цикл до выполнения условия abs(x2 - x1) < eps
    while abs(x2 - x1) > eps and k < max_iter:
        # матрица системы относительно alpha, beta, gamma
        # строки соответствуют уравнениям alpha + beta*xi + gamma*xi**2 = yi
        A = np.array([
            [1.0, x0, x0 * x0],
            [1.0, x1, x1 * x1],
            [1.0, x2, x2 * x2]
        ], dtype=float)
        B = np.array([y0, y1, y2], dtype=float)
        try:
            # решаем систему — находим коэффициенты параболы
            alpha, beta, gamma = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            break  # вырожденная матрица — выход
        if abs(gamma) < 1e-15:
            break  # парабола выродилась в прямую
        # вершина параболы: f'(x) = beta + 2*gamma*x = 0 → x3 = -beta/(2*gamma)
        x3 = -beta / (2.0 * gamma)
        # ограничиваем x3 внутри текущего интервала [x0, x2]
        lo, hi = min(x0, x2), max(x0, x2)
        if x3 < lo or x3 > hi:
            x3 = (x0 + x2) / 2.0
        # исключаем начальную точку и включаем новую
        x0, y0 = x1, y1
        x1, y1 = x2, y2
        x2, y2 = x3, f(x3)
        k += 1
    return x2, y2, k


def find_extrema_brackets(f, a, b, num=2000):
    # сканирует отрезок и находит отрезки с локальными экстремумами
    xs = np.linspace(a, b, num)
    ys = np.array([f(x) for x in xs])
    brackets = []
    for i in range(1, len(xs) - 1):
        # локальный минимум: точка ниже соседей
        if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]:
            brackets.append(((xs[i - 1], xs[i + 1]), 'min'))
        # локальный максимум: точка выше соседей
        elif ys[i] > ys[i - 1] and ys[i] > ys[i + 1]:
            brackets.append(((xs[i - 1], xs[i + 1]), 'max'))
    return brackets


def nelder_mead(f, x0, eps=1e-6, max_iter=1000):
    # метод Нелдера–Мида
    # симплекс из n+1 вершины, обозначения B (best), G (good), W (worst)
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    # строим начальный симплекс: x0 и n смещённых вершин
    simplex = [x0]
    for i in range(n):
        p = x0.copy()
        # смещение по i-й координате
        p[i] += 0.5 if abs(x0[i]) < 1e-12 else 0.05 * abs(x0[i])
        simplex.append(p)
    simplex = np.array(simplex)
    f_vals = np.array([f(p) for p in simplex])
    it = 0
    for it in range(max_iter):
        # упорядочиваем: simplex[0] = B (best), simplex[-1] = W (worst)
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]
        # критерий сходимости: симплекс сжался
        if np.max(np.abs(simplex[1:] - simplex[0])) < eps:
            break
        # B = simplex[0], G = simplex[-2], W = simplex[-1]
        B, G, W = simplex[0], simplex[-2], simplex[-1]
        # средняя точка хорошей стороны M = (B + G)/2
        M = (B + G) / 2.0
        # отражение R = 2M - W
        R = 2.0 * M - W
        fR = f(R)
        fB, fG, fW = f_vals[0], f_vals[-2], f_vals[-1]

        if fR < fG:
            # случай (i): отражение или растягивание
            if fB < fR:
                # f(B) < f(R) < f(G) — просто заменяем W на R
                simplex[-1], f_vals[-1] = R, fR
            else:
                # f(R) ≤ f(B) — пробуем растягивание
                E = 2.0 * R - M  # точка растягивания
                fE = f(E)
                if fE < fB:
                    # растягивание удачно — W := E
                    simplex[-1], f_vals[-1] = E, fE
                else:
                    # иначе W := R
                    simplex[-1], f_vals[-1] = R, fR
        else:
            # случай (ii): сжатие или сокращение
            if fR < fW:
                # f(G) <= f(R) < f(W) — заменяем W на R
                simplex[-1], f_vals[-1] = R, fR
            else:
                # сжатие: C = (W + M)/2 или C = (M + R)/2
                C1 = (W + M) / 2.0
                C2 = (M + R) / 2.0
                fC1 = f(C1)
                fC2 = f(C2)
                # выбираем точку с меньшим значением функции
                if fC1 < fC2:
                    C, fC = C1, fC1
                else:
                    C, fC = C2, fC2
                if fC < fW:
                    # сжатие удачно — W := C
                    simplex[-1], f_vals[-1] = C, fC
                else:
                    # сокращение по направлению к B
                    # G := M, W := S, где S = (B + W)/2
                    S = (B + W) / 2.0
                    simplex[-2], f_vals[-2] = M, f(M)
                    simplex[-1], f_vals[-1] = S, f(S)
    best = simplex[0]
    return best, f(best), it


def plot_function(f, a, b, extrema, title):
    # строит график функции и отмечает найденные экстремумы
    xs = np.linspace(a, b, 1000)
    ys = [f(x) for x in xs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, 'b-', label='f(x)', zorder=2)
    draw_axes(ax)

    first = True
    for (x, y, typ) in extrema:
        # красный кружок для минимума, зелёный крестик для максимума
        if typ == 'min':
            color, marker = 'ro', 'o'
        else:
            color, marker = 'gx', 'x'
        ax.plot(x, y, color, markersize=10, zorder=5,
                label=('минимум' if typ == 'min' else 'максимум') if first else None)
        first = False

    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.show()


def solve_direct_golden(f, a, b, eps, expr):
    # метод золотого сечения без отделения — надо указать min или max
    print("\nЧто искать?")
    print("1 - минимум")
    print("2 - максимум")
    t = input("Ваш выбор: ").strip()

    if t == '1':
        f_work = f
        typ = 'min'
    elif t == '2':
        f_work = lambda x, _f=f: -_f(x)
        typ = 'max'
    else:
        print("Неверный выбор.")
        return

    x, y_work, k = golden_section(f_work, a, b, eps)
    y = y_work if typ == 'min' else -y_work

    print(f"\nметод золотого сечения без отделения:")
    print(f"  x* = {x:.10f}, f(x*) = {y:.10f}")
    print(f"  тип: {typ}, итераций = {k}")

    plot_function(f, a, b, [(x, y, typ)], f'Золотое сечение ({typ})')


def solve_direct_brent(f, a, b, eps, expr):
    # метод Брента без отделения — он находит ближайший экстремум,
    # тип определяется автоматически по второй производной
    x, y, k = brent(f, a, b, eps)
    typ = detect_extremum_type(f, x)

    print(f"\nметод Брента без отделения:")
    print(f"  x* = {x:.10f}, f(x*) = {y:.10f}")
    print(f"  тип найденного экстремума: {typ}, итераций = {k}")
    print(f"  (метод Брента всегда ищет ближайший экстремум,")
    print(f"   тип определяется автоматически)")

    plot_function(f, a, b, [(x, y, typ)], f'Метод Брента ({typ})')


def solve_with_separation(f, a, b, method, eps, expr, h=0.05):
    # отделяет экстремумы на отрезке, потом уточняет каждый выбранным методом
    brackets = find_extrema_brackets(f, a, b, num=int((b - a) / h) + 1)

    if not brackets:
        print("\nлокальные экстремумы не найдены")
        return

    if method == 'brent':
        name = 'метод брента'
    else:
        name = 'метод золотого сечения'

    print(f"\n{name} с отделением:")
    extrema = []

    for (br, typ) in brackets:
        # для максимума минимизируем -f
        if typ == 'min':
            f_work = f
        else:
            f_work = lambda x, _f=f: -_f(x)

        if method == 'brent':
            x, y_neg, k = brent(f_work, br[0], br[1], eps)
        else:
            x, y_neg, k = golden_section(f_work, br[0], br[1], eps)

        # восстанавливаем знак для максимума
        y = y_neg if typ == 'min' else -y_neg

        print(f"  {typ}: x* = {x:.10f}, f(x*) = {y:.10f}, итераций = {k}")
        extrema.append((x, y, typ))

    plot_function(f, a, b, extrema, f'{name} с отделением')


def solve_multi(f, x0, expr, eps):
    # многомерная оптимизация методом Нелдера–Мида
    print(f"\nФункция: f = {expr}")
    print(f"Начальная точка: {x0}, eps = {eps}")
    xopt, fopt, it = nelder_mead(f, x0, eps)
    print(f"\n  x* = {np.array2string(xopt, precision=8)}")
    print(f"  f(x*) = {fopt:.8f}")
    print(f"  Итераций = {it}")
    n = len(x0)
    if n == 2:
        # для двумерного случая строим линии уровня
        xs = np.linspace(-4, 4, 200)
        ys = np.linspace(-4, 4, 200)
        Xg, Yg = np.meshgrid(xs, ys)
        Zg = np.zeros_like(Xg)
        for i in range(Xg.shape[0]):
            for j in range(Xg.shape[1]):
                Zg[i, j] = f([Xg[i, j], Yg[i, j]])
        fig, ax = plt.subplots(figsize=(7, 6))
        cs = ax.contour(Xg, Yg, Zg, levels=20, cmap='viridis')
        ax.plot(xopt[0], xopt[1], 'ro', markersize=8, label='минимум')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(expr)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        fig.colorbar(cs, ax=ax)
        plt.show()


def main():
    # спрашиваем имя файла с исходными данными
    filename = input("Введите имя файла с исходными данными: ").strip()
    data = read_input_file(filename)
    if data is None:
        return

    eps = data['eps']

    # если задана функция одной переменной — одномерная задача
    if data['function'] is not None:
        if data['segment'] is None:
            print("Для функции одной переменной нужен segment")
            return

        a, b = data['segment']
        expr = data['function']
        f = make_single_func(expr)

        print(f"\nФункция: y = {expr}")
        print(f"Отрезок: [{a}, {b}], eps = {eps}")

        # меню для одномерной задачи
        while True:
            print("\nМеню:")
            print("1 - Отделять экстремумы на отрезке")
            print("2 - Решать без отделения (на заданном отрезке)")
            print("0 - Выход")

            choice = input("Выберите пункт: ").strip()

            if choice in ('1', '2'):
                separate = (choice == '1')

                # шаг отделения спрашиваем только при отделении
                h = 0.05
                if separate:
                    h_input = input("Введите шаг отделения h (например, 0.05): ").strip()
                    h = float(h_input) if h_input else 0.05

                # выбор метода
                print("\nВыберите метод:")
                print("1 - Метод золотого сечения")
                print("2 - Метод Брента")
                m = input("Ваш выбор: ").strip()

                if m not in ('1', '2'):
                    print("Неверный метод.")
                    continue

                method = 'golden' if m == '1' else 'brent'

                eps_input = input("Введите eps (например, 1e-6): ").strip()
                eps_use = float(eps_input) if eps_input else eps

                if separate:
                    # при отделении тип каждого экстремума определяется автоматически
                    solve_with_separation(f, a, b, method, eps_use, expr, h)
                else:
                    # без отделения:
                    # для золотого сечения спрашиваем min/max
                    # для Брента — не спрашиваем, он сам найдёт ближайший экстремум
                    if method == 'golden':
                        solve_direct_golden(f, a, b, eps_use, expr)
                    else:
                        solve_direct_brent(f, a, b, eps_use, expr)

            elif choice == '0':
                print("Выход...")
                break
            else:
                print("Неверный пункт меню.")

    # иначе если задана многомерная функция — многомерная задача
    elif data['multifunction'] is not None:
        if data['initial'] is None:
            print("Для многомерной функции нужен initial")
            return
        f = make_multi_func(data['multifunction'])
        solve_multi(f, data['initial'], data['multifunction'], eps)


if __name__ == "__main__":
    main()