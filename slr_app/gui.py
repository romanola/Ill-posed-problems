import datetime
from time import sleep

import numpy as np
import scipy
from kivy.app import App
from kivy.storage.dictstore import DictStore
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang.builder import Builder
from matplotlib import pyplot as plt

Builder.load_file('slr.kv')

store = DictStore('storage.json')


def phillips(n):
    """
    Test problem: Phillips 'famous' problem
    Disctretization of the 'famous' first-kind Fredholm integral
    equation deviced by D. L. Phillips.  Define the function
    phi(x) = | 1 + cos(x*pi/3) ,  |x| <  3 .
             | 0               ,  |x| >= 3
    Then the kernel K, the solution f, and the right-hand side
    g are given by:
    K(s,t) = phi(s-t) ,
    f(t)   = phi(t) ,
    g(s)   = (6-|s|)*(1+.5*cos(s*pi/3)) + 9/(2*pi)*sin(|s|*pi/3) .
    Both integration intervals are [-6,6].
    The order n must be a multiple of 4.
    :param n: size of matrix A
    :return: A->matrix, b->vector, x->solution-vector
    """
    # Check input
    if n % 4 != 0:
        raise ValueError("The order n must be a multiple of 4")

    # Compute the matrix A
    h = 12 / n
    n4 = n // 4
    r1 = np.zeros(n)
    c = np.cos(np.arange(-1, n4 + 1) * 4 * np.pi / n)
    r1[:n4] = h + 9 / (h * np.pi ** 2) * (2 * c[1:n4 + 1] - c[:n4] - c[2:n4 + 2])
    r1[n4] = h / 2 + 9 / (h * np.pi ** 2) * (np.cos(4 * np.pi / n) - 1)
    A = scipy.linalg.toeplitz(r1)

    # Compute the right-hand side b
    b = np.zeros(n)
    c = np.pi / 3
    for i in range(int(n / 2), n):
        t1 = -6 + (i + 1) * h
        t2 = t1 - h

        b[i] = t1 * (6 - abs(t1) / 2) + ((3 - abs(t1) / 2) * np.sin(c * t1) - 2 / c * (np.cos(c * t1) - 1)) / c - t2 * (
                6 - abs(t2) / 2) - ((3 - abs(t2) / 2) * np.sin(c * t2) - 2 / c * (np.cos(c * t2) - 1)) / c
        b[int(n - i - 1)] = b[i]

    b = b / np.sqrt(h)

    # Compute the solution x
    x = np.zeros(n)
    a = (h + np.diff(np.sin(np.arange(0, (3 + 10 * np.finfo(float).eps), h)[:, np.newaxis] * c),
                     axis=0) / c) / np.sqrt(h)
    x[2 * n4:3 * n4] = a.ravel()
    x[n4:2 * n4] = x[3 * n4 - 1:2 * n4 - 1:-1]

    # print(x)

    return A, b, x


def err_phillips(A, b0, x, n, nl=0.005):
    """
    Calculate the error between solution vector x and
    solution vector xr from randomized least squares problem
    :param A: matrix, size=(n, n)
    :param b0: vector, size=(1, 4)
    :param x: solution-vector, size=(4, 1)
    :param n: size of matrix A
    :param nl: noise parameter
    :return: ||x - xr||^2
    """
    epsil = nl * np.random.randn(1, n)  # epsil - vector size=k, epsil_i - Normal(0, 1)
    # R = np.random.randn(n, n)
    R = np.random.normal(size=(n, n))
    b1 = b0 + epsil
    Rk = np.zeros((n, n))
    er_r = np.zeros(n)

    for k in range(n):
        Rk[k, :] = R[k, :]  # random matrix size=(k, n)
        xr = np.linalg.pinv(Rk[:k + 1] @ A) @ Rk[:k + 1] @ b1.T
        er_r[k] = np.linalg.norm(x - xr.T, 2) ** 2

        # Er_r[k] = er_r
    return er_r


def plot_err(err_list, nl):
    """
    plot results for 1 experiment
    :param k_list: sizes of k
    :param err_list: array of errors
    :param n: size of matrix A
    :param nl_list: array with different noise levels
    :return: None
    """
    err_min = (min(err_list), np.argmin(err_list) + 1)

    plt.plot(range(1, len(err_list) + 1), err_list, 'b-',
             label=f"n = {len(err_list)}, k = {err_min[1]}, nl = {nl}\nmin_err ={err_min[0]}")
    plt.plot(*err_min[::-1], 'ro')
    plt.xlabel('k')
    plt.ylabel('||x - xr||^2')
    plt.title(f'n = {len(err_list)}, min_err = {err_min[0]}, k = {err_min[1]}, nl = {nl}')
    plt.legend()
    name = f'plot{datetime.datetime.now()}.png'
    plt.savefig(name)
    plt.clf()  #comment to store the plot
    return name


class StartScreen(Screen):

    pass


class MainScreen(Screen):

    def calculate(self):
        try:
            nl = float(self.ids.nl.text)
            n = int(self.ids.n.text)
            A, b, x = phillips(n)
            err = err_phillips(A, b, x, n, nl)
            if store.exists('data'):
                store.delete('data')
            store.put('data', n=n, nl=nl, A=A, b=b, x=x, err=err)
            name = plot_err(err, nl)
            self.manager.screens[2].ids.plot.source = name
            self.manager.current = "results_screen"
        except Exception as e:
            print(e)


class ResultsScreen(Screen):
    pass


class SlrGuiApp(App):

    def build(self):
        sm = ScreenManager()
        sm.add_widget(StartScreen(name="start_screen"))
        sm.add_widget(MainScreen(name="main_screen"))
        sm.add_widget(ResultsScreen(name="results_screen"))

        return sm


if __name__ == '__main__':
    SlrGuiApp().run()
