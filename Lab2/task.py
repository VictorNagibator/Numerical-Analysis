import sys
import math

def read_matrix_from_file(filename):
    # Чтение матрицы A и вектора b из файла.
    # Формат:
    #    первая строка: n
    #    следующие n строк: матрица A
    #    следующая строка: вектор b
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


# LU-разложение 
def lu_decomposition(A):
    # LU-разложение матрицы A = L * U.
    # L — нижнетреугольная с единичной диагональю,
    # U — верхнетреугольная
    # Возвращает L, U
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    for i in range(n):
        # Вычисляем i-ю строку U
        for j in range(i, n):
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - s

        if abs(U[i][i]) < 1e-15:
            raise ZeroDivisionError(
                f"Нулевой ведущий элемент U[{i}][{i}]. "
                "LU-разложение без перестановок невозможно."
            )

        # Вычисляем i-й столбец L
        for j in range(i + 1, n):
            s = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - s) / U[i][i]

    return L, U


def lu_determinant(L, U):
    # Определитель матрицы A через LU-разложение
    n = len(L)
    det_val = 1.0
    for i in range(n):
        det_val *= L[i][i] * U[i][i]
    return det_val


def lu_solve(L, U, b):
    # Решение системы A x = b через LU-разложение
    # Прямая подстановка: L y = b
    # Обратная подстановка: U x = y
    n = len(b)

    # Прямая подстановка
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - s  # L[i][i] = 1

    # Обратная подстановка
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]

    return x, y


# Метод прогонки (для трёхдиагональной матрицы)
def tridiagonal_solve(a, b, c, d):
    # Решение системы с трёхдиагональной матрицей:
    #    a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i
    # a - поддиагональ, b - главная диагональ, c - наддиагональ, d - правая часть.
    # Возвращает x, alpha, beta (коэффициенты прогонки)
    n = len(d)
    alpha = [0.0] * n
    beta = [0.0] * n

    # Прямая прогонка
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * alpha[i - 1]
        if abs(denom) < 1e-15:
            raise ZeroDivisionError(f"Нулевой знаменатель на шаге прямой прогонки i={i}")
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom
        if i < n - 1:
            alpha[i] = -c[i] / denom
        else:
            alpha[i] = 0.0

    # Обратная прогонка
    x = [0.0] * n
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x, alpha, beta


def extract_tridiagonal(A, b):
    # Извлекает диагонали из полной матрицы A для метода прогонки
    # Проверяет, что матрица трёхдиагональная
    n = len(A)
    a = [0.0] * n  # поддиагональ (a[0] не используется)
    b_diag = [0.0] * n
    c = [0.0] * n  # наддиагональ (c[n-1] не используется)
    d = b[:]

    for i in range(n):
        b_diag[i] = A[i][i]
        if i > 0:
            a[i] = A[i][i - 1]
        if i < n - 1:
            c[i] = A[i][i + 1]

    # Проверка на трёхдиагональность
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(A[i][j]) > 1e-12:
                raise ValueError("Матрица не является трёхдиагональной")

    return a, b_diag, c, d


def main():
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

    print("\nИсходная система:")
    print_matrix(A, "Матрица A:")
    print_vector(b, "Вектор b:")

    print("\nВыберите метод решения:")
    print("1 - LU-разложение")
    print("2 - Метод прогонки (для трёхдиагональной матрицы)")
    choice_method = input("Ваш выбор (1/2): ").strip()

    if choice_method == '1':
        print("\nLU-разложение")

        try:
            L, U = lu_decomposition(A)
        except ZeroDivisionError as e:
            print(f"Ошибка: {e}")
            return

        print_matrix(L, "Матрица L:")
        print_matrix(U, "Матрица U:")

        # Проверка L * U = A
        n = len(A)
        LU = [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        print_matrix(LU, "Проверка L * U:")

        det_A = lu_determinant(L, U)
        print(f"\nОпределитель матрицы A: det(A) = {det_A:.10f}")

        x, y = lu_solve(L, U, b)
        print_vector(y, "Промежуточный вектор y (L y = b):")
        print_vector(x, "Решение системы x:")

        # Невязка
        r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        print_vector(r, "Невязка r = b - A*x:")

    elif choice_method == '2':
        print("\nМетод прогонки")

        try:
            a, b_diag, c, d = extract_tridiagonal(A, b)
        except ValueError as e:
            print(f"Ошибка: {e}")
            return

        print_vector(a, "Поддиагональ a:")
        print_vector(b_diag, "Главная диагональ b:")
        print_vector(c, "Наддиагональ c:")
        print_vector(d, "Правая часть d:")

        try:
            x, alpha, beta = tridiagonal_solve(a, b_diag, c, d)
        except ZeroDivisionError as e:
            print(f"Ошибка: {e}")
            return

        print_vector(alpha, "Прогоночные коэффициенты alpha:")
        print_vector(beta, "Прогоночные коэффициенты beta:")
        print_vector(x, "Решение системы x:")

        # Невязка
        n = len(A)
        r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        print_vector(r, "Невязка r = b - A*x:")

    else:
        print("Неверный выбор метода.")


if __name__ == "__main__":
    main()