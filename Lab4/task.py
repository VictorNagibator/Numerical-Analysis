import os
import sys
import numpy as np


# Читает матрицу из текстового файла и возвращает numpy-массив
def read_matrix_from_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            # отбрасываем комментарии, начинающиеся с '#'
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            # приводим все возможные разделители к пробелам
            line = line.replace(",", " ").replace(";", " ")
            parts = line.split()
            try:
                row = [float(p) for p in parts]
            except ValueError as e:
                raise ValueError(
                    f"Не удалось разобрать строку {line_no}: '{line}' ({e})"
                )
            rows.append(row)

    if not rows:
        raise ValueError("Файл не содержит данных.")

    # все строки должны иметь одинаковую длину
    n_cols = len(rows[0])
    for i, r in enumerate(rows, 1):
        if len(r) != n_cols:
            raise ValueError(
                f"Строка {i} имеет длину {len(r)}, ожидалось {n_cols}."
            )

    return np.array(rows, dtype=float)


# Степенной метод для частичной проблемы собственных значений

# Ищет максимальное по модулю собственное число и соответствующий
# собственный вектор. На каждой итерации вектор нормируется, чтобы
# избежать переполнения или обнуления.

# Возвращает пару (lambda_max, eigenvector)
def power_method(A, tol=1e-12, max_iter=10000, verbose=True):
    n = A.shape[0]
    # начальный вектор можно брать произвольным, лишь бы он не был нулевым
    x = np.ones(n, dtype=float)
    x = x / np.linalg.norm(x)
    lam_old = 0.0

    for it in range(1, max_iter + 1):
        y = A @ x
        # отношение Рэлея даёт приближение к собственному числу
        lam_new = float(np.dot(y, x))
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-300:
            raise RuntimeError("Вектор обратился в ноль — метод расходится.")
        x_new = y / norm_y

        if verbose and it % 50 == 0:
            print(f"  итерация {it:5d}: lambda ≈ {lam_new:.12f}")

        if abs(lam_new - lam_old) < tol:
            if verbose:
                print(f"  сошлось за {it} итераций.")
            return lam_new, x_new

        lam_old = lam_new
        x = x_new

    if verbose:
        print("  предупреждение: достигнут максимум итераций, результат приближённый.")
    return lam_old, x


# Метод Якоби (вращений) для полной проблемы собственных значений
# симметричной матрицы

# На каждом шаге обнуляется максимальный по модулю внедиагональный
# элемент с помощью ортогонального вращения. Собственные векторы
# накапливаются в столбцах матрицы V

# Возвращает (eigenvalues, eigenvectors)
def jacobi_eigen(A, tol=1e-12, max_iter=2000, verbose=True):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("Метод Якоби применим только к симметричным матрицам.")

    V = np.eye(n)
    it = 0
    for it in range(1, max_iter + 1):
        # находим максимальный внедиагональный элемент
        off = A - np.diag(np.diag(A))
        p, q = np.unravel_index(np.argmax(np.abs(off)), off.shape)
        if p == q or abs(A[p, q]) < tol:
            break

        if p > q:
            p, q = q, p

        a_pp = A[p, p]
        a_qq = A[q, q]
        a_pq = A[p, q]

        # параметры вращения через tau = ctg(2*theta)
        tau = (a_qq - a_pp) / (2.0 * a_pq)
        if tau >= 0:
            t = 1.0 / (tau + np.sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + np.sqrt(1.0 + tau * tau))
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t * c

        # обновляем элементы p-й и q-й строк (и столбцов)
        for i in range(n):
            if i != p and i != q:
                a_ip = A[i, p]
                a_iq = A[i, q]
                A[i, p] = c * a_ip - s * a_iq
                A[p, i] = A[i, p]
                A[i, q] = s * a_ip + c * a_iq
                A[q, i] = A[i, q]

        # диагональные элементы и обнуляемый внедиагональный
        A[p, p] = c * c * a_pp - 2.0 * s * c * a_pq + s * s * a_qq
        A[q, q] = s * s * a_pp + 2.0 * s * c * a_pq + c * c * a_qq
        A[p, q] = A[q, p] = 0.0

        # накапливаем собственные векторы
        for i in range(n):
            v_ip = V[i, p]
            v_iq = V[i, q]
            V[i, p] = c * v_ip - s * v_iq
            V[i, q] = s * v_ip + c * v_iq

        if verbose and it % 20 == 0:
            off_norm = np.linalg.norm(A - np.diag(np.diag(A)))
            print(f"  итерация {it:5d}: ||off|| = {off_norm:.3e}")

    if verbose:
        print(f"  сошлось за {it} итераций.")

    eigvals = np.diag(A).copy()
    # сортируем собственные числа по убыванию
    idx = np.argsort(-eigvals)
    return eigvals[idx], V[:, idx]


# Приводит матрицу A к верхней форме Хессенберга с помощью отражений
# Хаусхолдера. Возвращает пару (H, Q), где A = Q H Q^T
def hessenberg(A, verbose=False):
    n = A.shape[0]
    H = A.copy().astype(float)
    Q = np.eye(n)

    for k in range(n - 2):
        x = H[k + 1:, k].copy()
        nx = np.linalg.norm(x)
        if nx < 1e-15:
            continue

        # строим вектор отражения, обнуляющий все компоненты x, кроме первой
        e = np.zeros_like(x)
        e[0] = 1.0
        alpha = -np.sign(x[0]) * nx if x[0] != 0 else -nx
        v = x - alpha * e
        nv = np.linalg.norm(v)
        if nv < 1e-15:
            continue
        v = v / nv

        # применяем отражение слева и справа
        H[k + 1:, k:] -= 2.0 * np.outer(v, v @ H[k + 1:, k:])
        H[:, k + 1:] -= 2.0 * np.outer(H[:, k + 1:] @ v, v)
        # накапливаем матрицу Q
        Q[:, k + 1:] -= 2.0 * np.outer(Q[:, k + 1:] @ v, v)

        if verbose:
            print(f"  шаг {k + 1}: ||H[k+1:,k]|| = {np.linalg.norm(H[k+1:, k]):.3e}")

    return H, Q


# Вычисляет параметры вращения Гивенса (c, s) для пары чисел (a, b)

# Вращение Гивенса имеет вид G = [[c, -conj(s)], [s, conj(c)]]
# и подбирается так, чтобы G^H @ [a; b] = [r; 0], где r = sqrt(|a|^2+|b|^2)
def givens_params(a, b):
    r = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
    if r < 1e-300:
        return 1.0 + 0.0j, 0.0 + 0.0j
    return a / r, b / r


# Применяет вращение Гивенса слева к строкам i, i+1 матрицы M
# (умножение на G^H)
def apply_givens_left(M, i, c, s):
    for j in range(M.shape[1]):
        m_ij = M[i, j]
        m_i1j = M[i + 1, j]
        M[i, j] = np.conj(c) * m_ij + np.conj(s) * m_i1j
        M[i + 1, j] = -s * m_ij + c * m_i1j


# Применяет вращение Гивенса справа к столбцам i, i+1 матрицы M
# (умножение на G). Используется для накопления унитарной матрицы Q
def apply_givens_right(M, i, c, s):
    for j in range(M.shape[0]):
        m_ji = M[j, i]
        m_ji1 = M[j, i + 1]
        M[j, i] = m_ji * c + m_ji1 * s
        M[j, i + 1] = -m_ji * np.conj(s) + m_ji1 * np.conj(c)


# QR-разложение верхней хессенберговой матрицы вращениями Гивенса

# Так как у хессенберговой матрицы под главной диагональю находится
# только одна ненулевая диагональ, достаточно n-1 вращения Гивенса
# для обнуления всех поддиагональных элементов

# Возвращает (Q, R), где H = Q @ R
def qr_decompose_hessenberg(H):
    n = H.shape[0]
    R = H.astype(complex).copy()
    Q = np.eye(n, dtype=complex)

    for i in range(n - 1):
        c, s = givens_params(R[i, i], R[i + 1, i])
        apply_givens_left(R, i, c, s)   # обнуляем R[i+1, i]
        apply_givens_right(Q, i, c, s)  # накапливаем Q = Q @ G_i

    return Q, R


# Находит собственный вектор верхней треугольной матрицы T,
# соответствующий собственному числу T[k, k]

# Согласно лекции, собственный вектор определяется решением
# системы (T - lambda E) y = 0. Для треугольной матрицы это
# сводится к обратной подстановке: свободная компонента y[k] = 1,
# остальные вычисляются из уравнений строк снизу вверх
def eigenvector_of_triangular(T, k):
    n = T.shape[0]
    y = np.zeros(n, dtype=complex)
    y[k] = 1.0

    for i in range(k - 1, -1, -1):
        s = 0.0 + 0.0j
        for j in range(i + 1, k + 1):
            s += T[i, j] * y[j]
        d = T[i, i] - T[k, k]
        # при близких или кратных собственных числах компонента обнуляется
        if abs(d) < 1e-12:
            y[i] = 0.0
        else:
            y[i] = -s / d

    nrm = np.linalg.norm(y)
    if nrm > 1e-300:
        y = y / nrm
    return y


# QR-метод со сдвигами Уилкинсона для полной проблемы собственных
# значений произвольной матрицы

# Сначала матрица приводится к форме Хессенберга (преобразования
# Хаусхолдера), затем выполняются QR-итерации со сдвигами. QR-разложение
# на каждом шаге делается вращениями Гивенса, а собственные векторы
# находятся обратной подстановкой в треугольной матрице

# Собственные числа отделяются по одному (дефляция): как только
# поддиагональный элемент становится малым, соответствующее собственное
# число фиксируется, а рабочий блок уменьшается
def qr_algorithm(A, tol=1e-12, max_iter=5000, verbose=True):
    n = A.shape[0]

    if verbose:
        print("  приведение к форме Хессенберга...")
    H_real, Q0_real = hessenberg(A, verbose=verbose)

    # переходим в комплексную арифметику
    H = H_real.astype(complex)
    V = Q0_real.astype(complex)

    eigvals = np.zeros(n, dtype=complex)
    i = n - 1
    it = 0

    while i >= 0 and it < max_iter:
        it += 1
        if i == 0:
            eigvals[i] = H[i, i]
            break

        # критерий сходимости поддиагонального элемента
        if abs(H[i, i - 1]) < tol * (abs(H[i - 1, i - 1]) + abs(H[i, i]) + 1e-300):
            eigvals[i] = H[i, i]
            if verbose:
                print(f"  отделено собственное число #{i + 1}: {eigvals[i]:.6f}")
            i -= 1
            continue

        # сдвиг Уилкинсона по нижнему блоку 2x2
        a, b = H[i - 1, i - 1], H[i - 1, i]
        c, d = H[i, i - 1], H[i, i]
        tr = a + d
        det = a * d - b * c
        disc = np.sqrt(tr * tr - 4.0 * det)
        mu1 = (tr + disc) / 2.0
        mu2 = (tr - disc) / 2.0
        mu = mu1 if abs(mu1 - d) < abs(mu2 - d) else mu2

        # QR-шаг по ведущему блоку размера m x m
        m = i + 1
        Q, R = qr_decompose_hessenberg(H[:m, :m] - mu * np.eye(m, dtype=complex))
        H[:m, :m] = R @ Q + mu * np.eye(m, dtype=complex)
        V[:, :m] = V[:, :m] @ Q

        # численная стабилизация: обнуляем элементы ниже первой поддиагонали
        for i1 in range(m):
            for j1 in range(i1 - 1):
                H[i1, j1] = 0.0

        if verbose and it % 50 == 0:
            print(f"  итерация {it:5d}: осталось отделить {i + 1}")

    # остаток (если не сошлось за max_iter) заполняем диагональю
    for j in range(i + 1):
        eigvals[j] = H[j, j]

    # собственные векторы через обратную подстановку
    eigvecs = np.zeros((n, n), dtype=complex)
    for k in range(n):
        y = eigenvector_of_triangular(H, k)
        eigvecs[:, k] = V @ y

    # сортируем по убыванию модуля собственного числа
    idx = np.argsort(-np.abs(eigvals))
    return eigvals[idx], eigvecs[:, idx]


# Печатает невязку ||A v - lambda v|| для каждой найденной пары
def check_result(A, eigvals, eigvecs, name="метод"):
    print(f"\n  проверка ({name}):")
    A = np.array(A, dtype=complex)
    max_res = 0.0
    for k in range(len(eigvals)):
        v = eigvecs[:, k]
        nv = np.linalg.norm(v)
        if nv < 1e-300:
            continue
        v = v / nv
        res = np.linalg.norm(A @ v - eigvals[k] * v)
        max_res = max(max_res, res)
        print(f"   lambda_{k + 1} = {eigvals[k]:>+.10f}    ||A v - lambda v|| = {res:.3e}")
    print(f"   максимальная невязка: {max_res:.3e}")


# Печатает матрицу в удобочитаемом виде
def print_matrix(A, title="Матрица"):
    print(f"\n{title} ({A.shape[0]}x{A.shape[1]}):")
    for row in A:
        print("  " + "  ".join(f"{v:>10.5f}" for v in row))


# Печатает собственные числа и соответствующие им векторы
def print_eigenpairs(eigvals, eigvecs):
    print("\nСобственные числа и собственные векторы:")
    for k in range(len(eigvals)):
        if np.isreal(eigvals[k]) and abs(eigvals[k].imag) < 1e-10:
            print(f"\n  lambda_{k + 1} = {eigvals[k].real:.10f}")
        else:
            print(f"\n  lambda_{k + 1} = {eigvals[k]:.6f}")
        v = eigvecs[:, k]
        parts = []
        for x in v:
            if abs(x.imag) < 1e-10:
                parts.append(f"{x.real:+.6f}")
            else:
                parts.append(f"{x.real:+.6f}{x.imag:+.6f}j")
        print(f"  v_{k + 1} = [{', '.join(parts)}]")


# Печатает меню выбора метода
def menu():
    print("\nВыберите метод:")
    print("  1 — степенной метод (максимальное по модулю собственное число)")
    print("  2 — метод Якоби (симметричная матрица)")
    print("  3 — QR-метод со сдвигами (произвольная матрица)")
    print("  4 — выполнить все методы и сравнить с numpy")
    print("  0 — выход")


# Точка входа: читает матрицу и запускает цикл меню
def main():
    # получаем путь к файлу либо из аргумента командной строки, либо из ввода
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Введите путь к файлу с матрицей: ").strip()
        if not path:
            print("Путь не задан. Выход.")
            return

    try:
        A = read_matrix_from_file(path)
    except Exception as e:
        print(f"Ошибка чтения матрицы: {e}")
        return

    print_matrix(A, "Исходная матрица")

    while True:
        menu()
        choice = input("Ваш выбор: ").strip()

        if choice == "0":
            print("Завершение работы.")
            break

        elif choice == "1":
            print("\nСтепенной метод.")
            try:
                lam, vec = power_method(A, verbose=True)
                print(f"\nмаксимальное по модулю собственное число: lambda = {lam:.10f}")
                print("соответствующий собственный вектор:")
                print(f"  v = [{', '.join(f'{x:+.6f}' for x in vec)}]")
                # нормируем знак так, чтобы наибольшая по модулю компонента была > 0
                if vec[np.argmax(np.abs(vec))] < 0:
                    vec = -vec
                print(f"  (с учётом знака: v = [{', '.join(f'{x:+.6f}' for x in vec)}])")
                res = np.linalg.norm(A @ vec - lam * vec)
                print(f"  невязка ||A v - lambda v|| = {res:.3e}")
            except Exception as e:
                print(f"Ошибка: {e}")

        elif choice == "2":
            print("\nМетод Якоби (вращений).")
            try:
                eigvals, eigvecs = jacobi_eigen(A, verbose=True)
                print_eigenpairs(eigvals, eigvecs)
                check_result(A, eigvals, eigvecs, "Якоби")
            except Exception as e:
                print(f"Ошибка: {e}")

        elif choice == "3":
            print("\nQR-метод со сдвигами.")
            try:
                eigvals, eigvecs = qr_algorithm(A, verbose=True)
                print_eigenpairs(eigvals, eigvecs)
                check_result(A, eigvals, eigvecs, "QR")
            except Exception as e:
                print(f"Ошибка: {e}")

        elif choice == "4":
            print("\nПоследовательный запуск всех методов.")

            print("\n[1/3] степенной метод.")
            try:
                lam, vec = power_method(A, verbose=False)
                print(f"  lambda_max = {lam:.10f}")
                print(f"  v_max = [{', '.join(f'{x:+.6f}' for x in vec)}]")
            except Exception as e:
                print(f"  ошибка: {e}")

            print("\n[2/3] метод Якоби.")
            try:
                if np.allclose(A, A.T, atol=1e-10):
                    eigvals, eigvecs = jacobi_eigen(A, verbose=False)
                    print_eigenpairs(eigvals, eigvecs)
                else:
                    print("  матрица не симметрична — метод Якоби пропущен.")
            except Exception as e:
                print(f"  ошибка: {e}")

            print("\n[3/3] QR-метод со сдвигами.")
            try:
                eigvals, eigvecs = qr_algorithm(A, verbose=False)
                print_eigenpairs(eigvals, eigvecs)
            except Exception as e:
                print(f"  ошибка: {e}")

            print("\nсравнение с библиотекой numpy.linalg.eig:")
            try:
                w, _ = np.linalg.eig(A)
                idx = np.argsort(-np.abs(w))
                w = w[idx]
                for k, val in enumerate(w):
                    print(f"   lambda_{k + 1} = {val:.10f}")
            except Exception as e:
                print(f"  ошибка: {e}")

        else:
            print("Неверный пункт меню. Попробуйте снова.")


if __name__ == "__main__":
    main()