import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

m = 1
k = 1
b = 0.5
A = np.array([[0, 1], [-k / m, -b / m]])
B = np.array([[0], [1 / m]])
C = np.array([1, 0])
D = np.array([0])

t = np.linspace(0, 5, 23)
noise = np.random.normal(0, 0.1, len(t))
noiseFnc = interp1d(t, noise)

#1.1
def model(t, y):
    x = np.array([y]).T
    dx = A @ x + B @ np.array([[1]])
    val=dx.T[0]
    return dx.T[0]

#1.4
#y = [x1,x2,x~1,x~2]
def modelOBS(t,y,l1,l2):
    x=np.array([[y[0]],[y[1]]])
    xDash=np.array([[y[2]],[y[3]]])
    dx=A@x+B@np.array([[1]])
    L=np.array([[l1],[l2]])
    doElki=C@x-C@xDash
    dxDash=A@xDash+B@np.array([[1]])+L*doElki
    val=np.array([dx[0,0],dx[1,0],dxDash[0,0],dxDash[1,0]])
    return val

def modelOBSNoise(t,y,l1,l2):
    x=np.array([[y[0]],[y[1]]])
    xDash=np.array([[y[2]],[y[3]]])
    dx=A@x+B@np.array([[1]])+noiseFnc(t)
    L=np.array([[l1],[l2]])
    doElki=C@x-C@xDash
    dxDash=A@xDash+B@np.array([[1]])+L*doElki
    val=np.array([dx[0,0],dx[1,0],dxDash[0,0],dxDash[1,0]])
    return val

def modelOBSNoisedInput(t,y,l1,l2):
    x=np.array([[y[0]],[y[1]]])
    xDash=np.array([[y[2]],[y[3]]])
    dx=A@x+B@np.array([[1+noiseFnc(t)]])
    L=np.array([[l1],[l2]])
    doElki=C@x-C@xDash
    dxDash=A@xDash+B@np.array([[1]])+L*doElki
    val=np.array([dx[0,0],dx[1,0],dxDash[0,0],dxDash[1,0]])
    return val
def findValueL(omega):
    lmbd=-omega
    l1=-2*lmbd-1/2
    l2=lmbd*lmbd-1-0.5*l1
    return np.array([l1,l2])

def checkObser(cal, n):
    if np.linalg.matrix_rank(cal) == n:
        return True
    else:
        return False


def zadanie(active):
    ###1.1
    res = solve_ivp(model, [0, 10], [0, 0], rtol=1e-10)
    u = res.y
    y = C @ np.array(res.y[:, 0])
    if True:
        plt.figure('System')
        plt.plot(res.t, res.y[0], label='x1')
        plt.plot(res.t, res.y[1], label='x2')
        plt.legend()
    ###1.2
    obser = np.array([C, C @ A])
    print('System Observability -', checkObser(obser, 2))
    ###1.3
    #a) w_0 należy dobierać w taki sposób, by układ z obserwatorem był szybszy od sprzężenia
    #   im wyższa wartość w_0, tym szybciej układ będzie nadążał nad 'głównym układem'
    #b) nie, obserwator nie zadziała dla każdej wartości w_0, np. dla takich, które zdestabilizują układ
    ###1.4 jako modelOBS wyżej
    ###1.5
    # instrukcje warunkowe 'if True/False' mają służyć do wyłączania zbędnych wykresów
    # zerowe warunki początkowe:
    if True:
        initialConditions = [0, 0, 0, 0]
        timeFrame = [0, 10]
        # w=-1
        if True:
            l1 = findValueL(-1)
            resObs1 = solve_ivp(modelOBS, timeFrame, initialConditions, args=(l1), rtol=1e-10)
            plt.figure('System with Observer, w=-1, zero initial conditions')
            plt.plot(resObs1.t, resObs1.y[0], label='x1')
            plt.plot(resObs1.t, resObs1.y[1], label='x2')
            plt.plot(resObs1.t, resObs1.y[2], label='x~1')
            plt.plot(resObs1.t, resObs1.y[3], label='x~2')
            plt.legend()
        # w=1
        if True:
            l2 = findValueL(1)
            resObs2 = solve_ivp(modelOBS, timeFrame, initialConditions, args=(l2), rtol=1e-10)
            plt.figure('System with Observer, w=1, zero initial conditions')
            plt.plot(resObs2.t, resObs2.y[0], label='x1')
            plt.plot(resObs2.t, resObs2.y[1], label='x2')
            plt.plot(resObs2.t, resObs2.y[2], label='x~1')
            plt.plot(resObs2.t, resObs2.y[3], label='x~2')
            plt.legend()
        # w=5
        if True:
            l3 = findValueL(5)
            resObs3 = solve_ivp(modelOBS, timeFrame, initialConditions, args=(l3), rtol=1e-10)
            plt.figure('System with Observer, w=5, zero initial conditions')
            plt.plot(resObs3.t, resObs3.y[0], label='x1')
            plt.plot(resObs3.t, resObs3.y[1], label='x2')
            plt.plot(resObs3.t, resObs3.y[2], label='x~1')
            plt.plot(resObs3.t, resObs3.y[3], label='x~2')
            plt.legend()
        # w=10
        if True:
            l4 = findValueL(10)
            resObs4 = solve_ivp(modelOBS, timeFrame, initialConditions, args=(l4), rtol=1e-10)
            plt.figure('System with Observer, w=10, zero initial conditions')
            plt.plot(resObs4.t, resObs4.y[0], label='x1')
            plt.plot(resObs4.t, resObs4.y[1], label='x2')
            plt.plot(resObs4.t, resObs4.y[2], label='x~1')
            plt.plot(resObs4.t, resObs4.y[3], label='x~2')
            plt.legend()
    # x~(0) != x(0)
    if True:
        initialConditions = [0, 0, 3, 3]
        timeFrame = [0, 10]
        # w=-1
        if True:
            l1 = findValueL(-1)
            resObs1 = solve_ivp(modelOBS, timeFrame, initialConditions, args=(l1), rtol=1e-10)
            plt.figure('System with Observer, w=-1, initial conditions = [0, 0, 3, 3]')
            plt.plot(resObs1.t, resObs1.y[0], label='x1')
            plt.plot(resObs1.t, resObs1.y[1], label='x2')
            plt.plot(resObs1.t, resObs1.y[2], label='x~1')
            plt.plot(resObs1.t, resObs1.y[3], label='x~2')
            plt.legend()
        # w=1
        if True:
            l2 = findValueL(1)
            resObs2 = solve_ivp(modelOBS, timeFrame, initialConditions, args=(l2), rtol=1e-10)
            plt.figure('System with Observer, w=1, initial conditions = [0, 0, 3, 3]')
            plt.plot(resObs2.t, resObs2.y[0], label='x1')
            plt.plot(resObs2.t, resObs2.y[1], label='x2')
            plt.plot(resObs2.t, resObs2.y[2], label='x~1')
            plt.plot(resObs2.t, resObs2.y[3], label='x~2')
            plt.legend()
        # w=5
        if True:
            l3 = findValueL(5)
            resObs3 = solve_ivp(modelOBS, timeFrame, initialConditions, args=(l3), rtol=1e-10)
            plt.figure('System with Observer, w=5, initial conditions = [0, 0, 3, 3]')
            plt.plot(resObs3.t, resObs3.y[0], label='x1')
            plt.plot(resObs3.t, resObs3.y[1], label='x2')
            plt.plot(resObs3.t, resObs3.y[2], label='x~1')
            plt.plot(resObs3.t, resObs3.y[3], label='x~2')
            plt.legend()
        # w=10
        if True:
            l4 = findValueL(10)
            resObs4 = solve_ivp(modelOBS, timeFrame, initialConditions, args=(l4), rtol=1e-10)
            plt.figure('System with Observer, w=10, initial conditions = [0, 0, 3, 3]')
            plt.plot(resObs4.t, resObs4.y[0], label='x1')
            plt.plot(resObs4.t, resObs4.y[1], label='x2')
            plt.plot(resObs4.t, resObs4.y[2], label='x~1')
            plt.plot(resObs4.t, resObs4.y[3], label='x~2')
            plt.legend()
    # tylko dla w=-1 obserwator zachowuje się nieprawidzłowo (niestabilny)
    # dla wartości w>0 obserwator działa poprawnie
    # przy niezerowych wartościach początkowych obserwatora ten tym szybciej zaczyna nadążać za orginalnym sygnałem, im większa 'w'
    #1.6
    if True:
        initialConditions = [0, 0, 0, 0]
        timeFrame = [0, 5]
        l5 = findValueL(10)
        resObsN = solve_ivp(modelOBSNoise, timeFrame, initialConditions, args=(l5), rtol=1e-10)
        plt.figure('System with Observer and noised output, w=10')
        plt.plot(resObsN.t, resObsN.y[0], label='x1 Noise')
        plt.plot(resObsN.t, resObsN.y[1], label='x2 Noise')
        plt.plot(resObsN.t, resObsN.y[2], label='x~1 Noise')
        plt.plot(resObsN.t, resObsN.y[3], label='x~2 Noise')
        plt.legend()
        # obserwator dobrze radzi sobie z obserwacją zdefiniowanego wyjścia C=[1,0], ale gorzej ze stanem dla którego nie był skonfigurowany
        # im większa 'w' tym lepiej wygląda obserwacja sygnału wyjściowego x1, nie wpływa to jednak na jakość obserwacji x2 (sprawdzone 'ręcznie', nie zakodowano tego)
    #1.7
    if True:
        initialConditions = [0, 0, 0, 0]
        timeFrame = [0, 5]
        l6 = findValueL(10)
        resObsN = solve_ivp(modelOBSNoisedInput, timeFrame, initialConditions, args=(l6), rtol=1e-10)
        plt.figure('System with Observer and noised input, w=10')
        plt.plot(resObsN.t, resObsN.y[0], label='x1 Noise')
        plt.plot(resObsN.t, resObsN.y[1], label='x2 Noise')
        plt.plot(resObsN.t, resObsN.y[2], label='x~1 Noise')
        plt.plot(resObsN.t, resObsN.y[3], label='x~2 Noise')
        plt.legend()
        # tak, obserwator pozwala na estymację stanu, chociaż nie w pełni poprawną, zarówno kiedy sygnał sterujący zostanie zaszumiony, jak i kiedy zmienione...
        # zostaną parametry systemu
        # im większa 'w' tym mniejsze błędy (sprawdzone 'ręcznie', nie zakodowano tego)
    #1.8???
    if True:
        initialConditions = [0, 0, 0, 0]
        timeFrame = [0, 5]
        l7 = findValueL(10)
        resObs7 = solve_ivp(modelOBS, timeFrame, initialConditions, args=(l7), rtol=1e-10)
        x1=resObs7.y[0]
        x2=np.zeros(len(x1))
        Tp=5/len(x1)
        for i in range(len(x1)) :
            if i==0:
                x2[i]=x1[i]/Tp
            else:
                x2[i]=(x1[i]-x1[i-1])/Tp
        plt.figure('Luenberger and calculated observer')
        plt.plot(resObs7.t, resObs7.y[1], label='x2')
        plt.plot(resObs7.t, resObs7.y[3], label='x~2')
        plt.plot(resObs7.t, x2, label='Calculated observer')
        # estymator ten, jest w pewien sposób poprawny, ale jest gorszy od obserwatora Luenberger'a
        # to czy można rozszerzyć taką estymatę na większą ilość zmiennych stanu zależy od ich interpretacji i zależności między sobą
    plt.legend()
    plt.show()


if __name__ == '__main__':
    zadanie(True)
