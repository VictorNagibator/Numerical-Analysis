import sys
import math

def read_matrix_from_file(filename):
    # Чтение матрицы A и вектора b из файла
    # Формат: первая строка - размер n,
    #           затем n строк матрицы A,
    #           затем вектор b (одна строка с n числами)

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("Файл пуст")

    n = int(lines[0])
    A = []
    for i in range(1, n+1):
        row = list(map(float, lines[i].split()))
        if len(row) != n:
            raise ValueError(f"Строка {i} матрицы имеет неверную длину")
        A.append(row)
    if len(lines) < n+2:
        raise ValueError("Не хватает строки для вектора b")
    b = list(map(float, lines[n+1].split()))
    if len(b) != n:
        raise ValueError("Вектор b имеет неверную длину")
    return A, b

def read_matrix_from_stdin():
    # Ввод с клавиатуры

    n = int(input("Введите размерность системы n: "))
    print("Введите матрицу A построчно (через пробел):")
    A = []
    for i in range(n):
        row = list(map(float, input().split()))
        if len(row) != n:
            print("Ошибка: неверное количество элементов, попробуйте снова")
            return read_matrix_from_stdin()
        A.append(row)
    print("Введите вектор b (через пробел):")
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

def gaussian_elimination(A, b, tol=1e-12):
    # Метод Гаусса с выбором главного элемента в столбце
    # Возвращает решение x или None, если матрица вырождена
    
    n = len(A)
    # Создаём расширенную матрицу (копируем, чтобы не менять исходные значения)
    M = [A[i][:] + [b[i]] for i in range(n)]
    
    # Прямой ход
    for k in range(n):
        # Поиск главного элемента в столбце k (строки k..n-1)
        pivot = k
        max_abs = abs(M[k][k])
        for i in range(k+1, n):
            if abs(M[i][k]) > max_abs:
                max_abs = abs(M[i][k])
                pivot = i
        if max_abs < tol:
            # Матрица вырождена
            return None
        # Перестановка строк, если нужно
        if pivot != k:
            M[k], M[pivot] = M[pivot], M[k]
        
        # Исключение элементов под диагональю
        for i in range(k+1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n+1):
                M[i][j] -= factor * M[k][j]
    
    # Обратный ход
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        if abs(M[i][i]) < tol:
            return None
        s = sum(M[i][j] * x[j] for j in range(i+1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return x

def gauss_jordan_solve(A, b, tol=1e-12):
    # Метод Гаусса-Жордана для решения СЛАУ.
    # Приводит расширенную матрицу к единичному виду
    # Возвращает решение x или None

    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    
    for k in range(n):
        # Выбор главного элемента
        pivot = k
        max_abs = abs(M[k][k])
        for i in range(k+1, n):
            if abs(M[i][k]) > max_abs:
                max_abs = abs(M[i][k])
                pivot = i
        if max_abs < tol:
            return None
        if pivot != k:
            M[k], M[pivot] = M[pivot], M[k]
        
        # Нормировка строки k
        div = M[k][k]
        for j in range(k, n+1):
            M[k][j] /= div
        
        # Обнуление всех остальных строк в столбце k
        for i in range(n):
            if i == k:
                continue
            factor = M[i][k]
            for j in range(k, n+1):
                M[i][j] -= factor * M[k][j]
    
    # В последнем столбце теперь решение
    x = [M[i][n] for i in range(n)]
    return x

def gauss_jordan_inverse(A, tol=1e-12):
    # Вычисление обратной матрицы методом Гаусса-Жордана.
    # Возвращает обратную матрицу или None, если исходная вырождена.
    n = len(A)
    # Расширенная матрица [A | I]
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    for k in range(n):
        # Выбор главного элемента
        pivot = k
        max_abs = abs(M[k][k])
        for i in range(k+1, n):
            if abs(M[i][k]) > max_abs:
                max_abs = abs(M[i][k])
                pivot = i
        if max_abs < tol:
            return None
        if pivot != k:
            M[k], M[pivot] = M[pivot], M[k]
        
        # Нормировка строки k
        div = M[k][k]
        for j in range(2*n):  # все элементы строки
            M[k][j] /= div
        
        # Обнуление всех остальных строк в столбце k
        for i in range(n):
            if i == k:
                continue
            factor = M[i][k]
            for j in range(2*n):
                M[i][j] -= factor * M[k][j]
    
    # Извлекаем правую часть (последние n столбцов)
    inv = [M[i][n:] for i in range(n)]
    return inv

def main():
    print("Решение СЛАУ")
    print("1 - ввод с клавиатуры")
    print("2 - чтение из файла")
    choice = input("Выберите способ ввода (1/2): ")
    
    if choice == '2':
        filename = input("Введите имя файла: ")
        try:
            A, b = read_matrix_from_file(filename)
        except Exception as e:
            print(f"Ошибка чтения файла: {e}")
            return
    else:
        A, b = read_matrix_from_stdin()
    
    print("Исходная система:")
    print_matrix(A, "Матрица A")
    print_vector(b, "Вектор b")
    
    # 1. Метод Гаусса с выбором главного элемента
    print("\nМетод Гаусса с выбором главного элемента:")
    x_gauss = gaussian_elimination(A, b)
    if x_gauss is not None:
        print_vector(x_gauss, "Решение:")
        # Проверка невязки
        r = [b[i] - sum(A[i][j]*x_gauss[j] for j in range(len(A))) for i in range(len(A))]
        print_vector(r, "Невязка r = b - A*x")
    else:
        print("Матрица вырождена, решение не единственно или отсутствует")
    
    # 2. Метод Гаусса-Жордана для решения
    print("\nМетод Гаусса-Жордана (решение СЛАУ):")
    x_jordan = gauss_jordan_solve(A, b)
    if x_jordan is not None:
        print_vector(x_jordan, "Решение:")
        r = [b[i] - sum(A[i][j]*x_jordan[j] for j in range(len(A))) for i in range(len(A))]
        print_vector(r, "Невязка r = b - A*x")
    else:
        print("Матрица вырождена, решение не единственно или отсутствует")
    
    # 3. Вычисление обратной матрицы методом Гаусса-Жордана
    print("\nВычисление обратной матрицы методом Гаусса-Жордана:")
    inv = gauss_jordan_inverse(A)
    if inv is not None:
        print_matrix(inv, "Обратная матрица:")
        # Проверка: A * A_-1 = I
        n = len(A)
        E = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                E[i][j] = sum(A[i][k] * inv[k][j] for k in range(n))
        print_matrix(E, "A * A_-1 (должна быть единичной):")
    else:
        print("Матрица вырождена, обратной не существует")

if __name__ == "__main__":
    main()