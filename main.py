import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m = 1
k = 1
b = 0.5
A = np.array([[0, 1], [-k / m, -b / m]])
B = np.array([[0], [1 / m]])
C = np.array([1, 0])
D = np.array([0])

def model(t,y):
    x = np.array([y]).T
    dx = A @ x + B @ np.array([[1]])
    return dx.T[0]

def checkObser(cal,n):
    if np.linalg.matrix_rank(cal)==n:
        return True
    else:
        return False

def zadanie1(active):
    res = solve_ivp(model, [0, 10], [0, 0], rtol=1e-10)
    u=res.y
    y = C @ np.array(res.y[:,0])
    obser=np.array([C,C@A])
    print('System Observability -',checkObser(obser,2))
    t=np.linspace(0,5,101)
    szzz=np.random.normal(0,0.1,len(t))
    # plt.figure()
    # plt.plot(res.t, res.y[0])
    # plt.plot(res.t, res.y[1])
    # plt.show()
    pass

if __name__ == '__main__':
    zadanie1(True)
