import math
import numpy as np


def read_matrix_from_file(filename):
    # Чтение матрицы A и вектора b из файла
    # Формат: первая строка n; следующие n строк — A; следующая — b
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError("Файл пуст")
    n = int(lines[0])
    A = []
    for i in range(1, n + 1):
        row = list(map(float, lines[i].split()))
        if len(row) != n:
            raise ValueError(f"Строка {i} матрицы имеет неверную длину")
        A.append(row)
    if len(lines) < n + 2:
        raise ValueError("Не хватает строки для вектора b")
    b = list(map(float, lines[n + 1].split()))
    if len(b) != n:
        raise ValueError("Вектор b имеет неверную длину")
    return A, b


def read_matrix_from_stdin():
    # Ввод матрицы A и вектора b с клавиатуры
    n = int(input("Введите размерность системы n: "))
    print("Введите матрицу A построчно (элементы через пробел):")
    A = []
    for i in range(n):
        row = list(map(float, input().split()))
        if len(row) != n:
            print("Ошибка: неверное количество элементов, попробуйте снова")
            return read_matrix_from_stdin()
        A.append(row)
    print("Введите вектор b (элементы через пробел):")
    b = list(map(float, input().split()))
    if len(b) != n:
        print("Ошибка: неверное количество элементов")
        return read_matrix_from_stdin()
    return A, b


def print_matrix(mat, title="Матрица"):
    print(title)
    for row in mat:
        print(' '.join(f"{x:12.6f}" for x in row))


def print_vector(vec, title="Вектор"):
    print(title)
    print(' '.join(f"{x:12.6f}" for x in vec))


# Проверка диагонального преобладания
def check_diagonal_dominance(A):
    n = len(A)
    ok_all = True
    print("\nПроверка условия диагонального преобладания:")
    for i in range(n):
        diag = abs(A[i][i])
        off = sum(abs(A[i][j]) for j in range(n) if j != i)
        ok = diag > off
        print(f"  Строка {i + 1}: |{A[i][i]:>8.4f}| = {diag:>8.4f} "
              f"{'>' if ok else '<='} {off:>8.4f}  {'OK' if ok else 'НЕТ'}")
        if not ok:
            ok_all = False
    if ok_all:
        print("  => Условие ВЫПОЛНЕНО (сходимость гарантирована)")
    else:
        print("  => Условие НЕ ВЫПОЛНЕНО (сходимость не гарантирована)")
    return ok_all

# Матрица C и вектор d для схемы x = Cx + d
def build_iteration_matrices(A, b):
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    d = [0.0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i][j] = -A[i][j] / A[i][i]
        d[i] = b[i] / A[i][i]
    return C, d

# Вычисление невязки
def compute_residual(A, b, x):
    n = len(A)
    return [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]


# Вывод итераций
def print_iterations(iterations, eps, max_show=25):
    print(f"\n{'Итерация':<10}{'Невязка':<18}{'Приближение'}")
    print("-" * 80)
    show = iterations[:max_show] if len(iterations) > max_show else iterations
    for k, x, diff in show:
        x_str = '  '.join(f"{v:>10.6f}" for v in x)
        mark = "  <== точность достигнута" if diff < eps else ""
        print(f"{k:<10}{diff:<18.8f}{x_str}{mark}")
    if len(iterations) > max_show:
        print(f"... показаны первые {max_show} итераций из {len(iterations)}")


# Метод Якоби
def jacobi_solve(A, b, eps=1e-3, max_iter=1000, x0=None):
    n = len(A)
    if x0 is None:
        x0 = [0.0] * n
    C, d = build_iteration_matrices(A, b)
    x = x0[:]
    iterations = []
    for k in range(max_iter):
        x_new = [0.0] * n
        for i in range(n):
            s = d[i]
            for j in range(n):
                if j != i:
                    s += C[i][j] * x[j]
            x_new[i] = s
        diff = math.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n)))
        iterations.append((k + 1, x_new[:], diff))
        if diff < eps:
            return x_new, iterations, C, d
        x = x_new
    return x, iterations, C, d


# Метод Зейделя
def seidel_solve(A, b, eps=1e-3, max_iter=1000, x0=None):
    n = len(A)
    if x0 is None:
        x0 = [0.0] * n
    C, d = build_iteration_matrices(A, b)
    x = x0[:]
    iterations = []
    for k in range(max_iter):
        x_old = x[:]
        for i in range(n):
            s = d[i]
            for j in range(n):
                if j != i:
                    if j < i:
                        s += C[i][j] * x[j] # уже обновлённое
                    else:
                        s += C[i][j] * x_old[j] # старое
            x[i] = s
        diff = math.sqrt(sum((x[i] - x_old[i]) ** 2 for i in range(n)))
        iterations.append((k + 1, x[:], diff))
        if diff < eps:
            return x, iterations, C, d
    return x, iterations, C, d


# Проверка симметричности матрицы
def is_symmetric(A, tol=1e-9):
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True

def get_eigenvalue_bounds(A, verbose=True):
    # Возвращает (mu, Lambda) — минимальное и максимальное собственные числа A
    # Работает для симметричных матриц (через numpy.linalg.eigvalsh)
    # Возвращает (None, None), если матрица не положительно определённая (есть lambda <= 0)
    # или не симметрична

    if not is_symmetric(A):
        if verbose:
            print("  Матрица не симметрична -> формула tau = 2/(mu + Lambda) "
                  "неприменима")
        return None, None

    A_np = np.array(A, dtype=float)
    eig = np.linalg.eigvalsh(A_np) # для симметричной — вещественные
    mu, Lambda = float(eig[0]), float(eig[-1])

    if verbose:
        print(f"  Собственные числа A: {list(np.round(eig, 6))}")
        print(f"  mu = {mu:.6f}, Lambda = {Lambda:.6f}")

    if mu <= 0:
        if verbose:
            print("  mu <= 0 -> матрица не положительно определённая. Метод простой итерации "
                  "с tau = 2/(mu + Lambda) может не сойтись.")
        return None, None

    return mu, Lambda


# Метод простой итерации
def simple_iteration_solve(A, b, tau=None, eps=1e-3, max_iter=1000, x0=None):
    # x^(m+1) = x^(m) - tau*(A x^(m) - b)
    # Если tau не задан — выбирается оптимально:
    #     tau0 = 2 / (mu + Lambda),
    # где mu, Lambda — границы множества собственных значений A 
    # (для положительно определённой матрицы — точные lambda)
    n = len(A)
    if x0 is None:
        x0 = [0.0] * n

    if tau is None:
        print("\nПоиск границ собственных значений матрицы A для выбора tau:")
        mu, Lambda = get_eigenvalue_bounds(A)

        if mu is not None and Lambda is not None:
            tau = 2.0 / (mu + Lambda)
            print(f"  tau_opt = 2/(mu + Lambda) = 2/({mu:.4f} + {Lambda:.4f}) "
                  f"= {tau:.6f}")
        else:
            tau = 0.1
            print(f"  Не удалось корректно оценить собственные значения. "
                  f"Используем tau = {tau}. Метод может расходиться.")

    x = x0[:]
    iterations = []
    for k in range(max_iter):
        Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        x_new = [x[i] - tau * (Ax[i] - b[i]) for i in range(n)]
        diff = math.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n)))
        iterations.append((k + 1, x_new[:], diff))
        if diff < eps:
            return x_new, iterations, tau
        x = x_new
    return x, iterations, tau


def main():
    print("Решение СЛАУ итерационными методами (Якоби, Зейдель, простая итерация)")

    print("\nВыберите способ ввода данных:")
    print("1 - ввод с клавиатуры")
    print("2 - чтение из файла")
    choice_input = input("Ваш выбор (1/2): ").strip()

    if choice_input == '2':
        filename = input("Введите имя файла: ").strip()
        try:
            A, b = read_matrix_from_file(filename)
        except Exception as e:
            print(f"Ошибка чтения файла: {e}")
            return
    else:
        A, b = read_matrix_from_stdin()

    n = len(A)

    print("\nИсходная система:")
    print_matrix(A, "Матрица A:")
    print_vector(b, "Вектор b:")

    check_diagonal_dominance(A)

    # Точность
    eps_str = input("\nВведите точность eps (Enter = 0.001): ").strip()
    try:
        eps = float(eps_str) if eps_str else 1e-3
    except ValueError:
        eps = 1e-3
    print(f"Точность: eps = {eps}")

    # Начальное приближение
    print("\nВыберите начальное приближение:")
    print("1 - нулевой вектор")
    print("2 - вектор b_i / a_ii")
    print("3 - ввести вручную")
    choice_x0 = input("Ваш выбор (1/2/3): ").strip()

    if choice_x0 == '2':
        x0 = [b[i] / A[i][i] for i in range(n)]
    elif choice_x0 == '3':
        x0_str = input(f"Введите {n} чисел через пробел: ").strip()
        try:
            x0 = list(map(float, x0_str.split()))
            if len(x0) != n:
                raise ValueError
        except ValueError:
            print("Ошибка ввода, используем нулевой вектор")
            x0 = [0.0] * n
    else:
        x0 = [0.0] * n

    print(f"Начальное приближение x0 = {x0}")

    # Меню методов
    while True:
        print("\nВыберите метод решения:")
        print("1 - Метод Якоби")
        print("2 - Метод Зейделя")
        print("3 - Метод простой итерации")
        print("0 - Выход")
        choice_method = input("Ваш выбор: ").strip()

        if choice_method == '0':
            print("\nВыход...")
            break

        elif choice_method == '1':
            print("\nМетод Якоби")

            try:
                result, iterations, C, d = jacobi_solve(A, b, eps=eps, x0=x0)
            except OverflowError:
                print("\n[!] Метод расходится.")
                print("    Проверьте условие диагонального преобладания матрицы A.")
                continue

            print_matrix(C, "\nМатрица C:")
            print_vector(d, "Вектор d:")

            print_iterations(iterations, eps)

            print_vector(result, "\nРешение системы x:")
            print_vector(compute_residual(A, b, result), "Невязка r = b - A*x:")

        elif choice_method == '2':
            print("\nМетод Зейделя")

            try:
                result, iterations, C, d = seidel_solve(A, b, eps=eps, x0=x0)
            except OverflowError:
                print("\n[!] Метод расходится.")
                print("    Проверьте условие диагонального преобладания матрицы A.")
                continue

            print_matrix(C, "\nМатрица C:")
            print_vector(d, "Вектор d:")

            print_iterations(iterations, eps)

            print_vector(result, "\nРешение системы x:")
            print_vector(compute_residual(A, b, result), "Невязка r = b - A*x:")

        elif choice_method == '3':
            print("\nМетод простой итерации")

            print("\nСпособ задания параметра tau:")
            print("1 - автоматически (через границы собственных значений)")
            print("2 - вручную")
            choice_tau = input("Ваш выбор (1/2): ").strip()

            tau = None
            if choice_tau == '2':
                try:
                    tau = float(input("Введите tau: ").strip())
                except ValueError:
                    print("Ошибка, используем автоматический выбор")
                    tau = None

            result, iterations, tau_used = simple_iteration_solve(
                A, b, tau=tau, eps=eps, x0=x0)

            print(f"\nИспользован параметр tau = {tau_used:.6f}")

            # B = E - tau*A
            B = [[(1.0 if i == j else 0.0) - tau_used * A[i][j]
                  for j in range(n)] for i in range(n)]
            print_matrix(B, "Матрица B = E - tau*A:")

            print_iterations(iterations, eps)

            print_vector(result, "\nРешение системы x:")
            print_vector(compute_residual(A, b, result), "Невязка r = b - A*x:")

        else:
            print("Неверный выбор метода.")


if __name__ == "__main__":
    main()