import matplotlib as matplotlib
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline
import scipy
import scipy.sparse
from scipy.stats import norm
from sympy import Symbol, Matrix
from sympy.interactive import printing


def init_x():
    """ Initial State """
    x = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    print(x, x.shape)
    print('x and x.shape')
    return x


def init_P():
    """ Initial Uncertainty - Initial Covariance Matrix """
    # Systemrauschen
    P = np.matrix([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]])
    print(P, P.shape)
    print('P and P.shape')
    return P


def init_A(dt):
    """ Dynamic Matrix """
    A = np.matrix([[1.0, 0.0, dt, 0.0, 1 / 2.0 * dt ** 2, 0.0],
                   [0.0, 1.0, 0.0, dt, 0.0, 1 / 2.0 * dt ** 2],
                   [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    print(A, A.shape)
    print('A and A.shape')
    return A


def init_H():
    """ Measurement Matrix """
    H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    print(H, H.shape)
    print('H and H.shape')

    return H


def init_R():
    """ Measurement Noise Covariance Matrix """
    ra = 5.0 ** 2  # Noise of Acceleration Measurement
    rp = 5.0 ** 2  # Noise of Position Measurement
    # ra = 0.5 ** 2  # Noise of Acceleration Measurement
    # rp = 0.5 ** 2  # Noise of Position Measurement
    R = np.matrix([[rp, 0.0, 0.0, 0.0],
                   [0.0, rp, 0.0, 0.0],
                   [0.0, 0.0, ra, 0.0],
                   [0.0, 0.0, 0.0, ra]])
    print(R, R.shape)
    print('R and R.shape')
    return R


def init_As_Gs():
    """
    We can easily calcualte Q, one can ask the question: How the noise effects my state vector?
    For example, how the jerk change the position over one timestep dt.
    With as the magnitude of the standard deviation of the jerk, which disturbs the car.
    We do not assume cross correlation, which means if a jerk will act in x direction of the movement, it will not push in y direction at the same time.
    We can construct the values with the help of a matrix , which is an "actor" to the state vector.
    """
    printing.init_printing(use_latex=True)
    dts = Symbol('\Delta t')
    As = Matrix([[1, 0, dts, 0, 1 / 2 * dts ** 2, 0],
                 [0, 1, 0, dts, 0, 1 / 2 * dts ** 2],
                 [0, 0, 1, 0, dts, 0],
                 [0, 0, 0, 1, 0, dts],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]])
    # this
    Gs = Matrix([dts ** 3 / 6, dts ** 2 / 2, dts])
    Gs
    # Gs * Gs.T
    return As, Gs


def init_Q(dt):
    """ Process Noise Covariance Matrix Q for CA Model """
    sj = 0.1
    q_val = 0.01
    """
    Q = np.matrix([[(dt ** 6) / 36, 0, (dt ** 5) / 12, 0, (dt ** 4) / 6, 0],
                   [0, (dt ** 6) / 36, 0, (dt ** 5) / 12, 0, (dt ** 4) / 6],
                   [(dt ** 5) / 12, 0, (dt ** 4) / 4, 0, (dt ** 3) / 2, 0],
                   [0, (dt ** 5) / 12, 0, (dt ** 4) / 4, 0, (dt ** 3) / 2],
                   [(dt ** 4) / 6, 0, (dt ** 3) / 2, 0, (dt ** 2), 0],
                   [0, (dt ** 4) / 6, 0, (dt ** 3) / 2, 0, (dt ** 2)]]) * sj ** 2
    """
    Q = np.matrix([[q_val, 0, 0, 0, 0, 0],
              [0, q_val, 0, 0, 0, 0],
              [0, 0, q_val, 0, 0, 0],
              [0, 0, 0, q_val, 0, 0],
              [0, 0, 0, 0, q_val, 0],
              [0, 0, 0, 0, 0, q_val]])

    print(Q, Q.shape)
    print('Q and Q.shape')

    return Q


def init_I(n):
    """ Identity Matrix """
    I = np.eye(n)
    print(I, I.shape)
    print('I and I.shape')
    return I


def sim_measurement_pos(m, sp, px, py, dx):
    """ Simulate Measurement of Position Sensors with random values """
    mpx_noise = np.array(np.random.randn(m))
    # mpy = np.array(py + sp * np.random.randn(m))

    """ Simulate Measurement of Position Sensors with a sinus """
    # struggle to get the correct input array with np.arange and np.linspace
    # mpx_1Hz = np.arange(-1.0, 1.0, 0.01)
    #
    mpx_1Hz = np.empty([m], dtype=float)
    mpx_1Hz[0] = px
    mpx_10Hz = np.empty([m], dtype=float)
    mpx_10Hz[0] = py
    for n in range(m-1):
        mpx_1Hz[n + 1] = mpx_1Hz[n] + dx
        mpx_10Hz[n + 1] = mpx_10Hz[n] + dx
        """
        mpx_1Hz[n + 1] = mpx_1Hz[n] + (1/m * 10)
        mpx_10Hz[n + 1] = mpx_10Hz[n] + (1/m * 10)
        """

    mpy_1Hz = np.sin(mpx_1Hz) * 2.5
    mpy_10Hz = np.sin(mpx_10Hz) * 2.5
    """
    mpy_1Hz = np.sin(mpx_1Hz * 2) * 2.5
    mpy_10Hz = np.sin(mpx_10Hz * 2) * 2.5
    """

    # make a upward sinus
    for n in range(m-1):
        mpy_1Hz[n] = mpy_1Hz[n] + dx*n*0.2
        mpy_10Hz[n] = mpy_10Hz[n] + dx*n*0.2

    # make some noise onto the sinus for the kalman, change 0.0 to something else
    for n in range(m-1):
        mpy_1Hz[n] = mpy_1Hz[n] + mpx_noise[n] * 0.0

    # Generate POS Trigger
    POS = np.ndarray(m, dtype='bool')
    POS[0] = True
    # Less new position updates, every 10th
    for i in range(1, m):
        if i % 10 == 0:
            POS[i] = True
        else:
            mpx_1Hz[i] = mpx_1Hz[i - 1]
            mpy_1Hz[i] = mpy_1Hz[i - 1]
            POS[i] = False

    # generate some delay for the position measurement
    delta = 7
    mpx_1Hz_temp = np.empty([m], dtype=float)
    mpy_1Hz_temp = np.empty([m], dtype=float)
    mpx_1Hz_temp[(delta):(m)] = mpx_1Hz[0:(m-delta)]
    mpx_1Hz_temp[0:delta] = px
    mpy_1Hz_temp[(delta):(m)] = mpy_1Hz[0:(m-delta)]
    mpy_1Hz_temp[0:delta] = py

    mpx_1Hz = mpx_1Hz_temp
    mpy_1Hz = mpy_1Hz_temp

    return mpx_1Hz, mpy_1Hz, mpx_10Hz, mpy_10Hz, POS


def sim_measurement_ac(m, sa, ax, ay, mpx_1Hz, mpy_1Hz, mpx_10Hz, mpy_10Hz, dx):
    """ Simulate Measurement of Acceleration Sensors with random values """
    # mx = np.array(ax + sa * np.random.randn(m))
    # my = np.array(ay + sa * np.random.randn(m))

    """ Simulate Measurement of Acceleration Sensors with the input of the sinus position values """

    derivative1_x = np.gradient(mpx_10Hz, dx)
    derivative1_y = np.gradient(mpy_10Hz, dx)
    derivative2_x = np.gradient(derivative1_x, dx)
    derivative2_y = np.gradient(derivative1_y, dx)

    my = np.empty([m], dtype=float)
    mx = derivative2_x
    my[0:(m-4)] = derivative2_y[0:(m-4)]
    my[(m-4):(m)] = derivative2_y[(m-4)]

    measurements = np.vstack((mpx_1Hz, mpy_1Hz, mx, my))

    print(measurements.shape)
    print('measurement shape')
    print(measurements[:, 1])
    print('(measurements[:, 1]')

    return mx, my, measurements


def kalman(x, P, A, H, R, Q, I, measurements, POS):
    # Preallocation for Plotting
    xt = []
    yt = []
    dxt = []
    dyt = []
    ddxt = []
    ddyt = []
    Zx = []
    Zy = []
    Px = []
    Py = []
    Pdx = []
    Pdy = []
    Pddx = []
    Pddy = []
    Kx = []
    Ky = []
    Kdx = []
    Kdy = []
    Kddx = []
    Kddy = []

    for n in range(m):
        # Time Update (Prediction)
        # ========================
        # Project the state ahead
        x = A * x
        # Project the error covariance ahead
        P = A * P * A.T + Q
        # Measurement Update (Correction)
        # ===============================
        # Calculate the kalman position only if there is new position data
        if POS[n]:
            # Compute the Kalman Gain
            S = H * P * H.T + R
            K = (P * H.T) * np.linalg.pinv(S)
            # Update the estimate via z
            Z = measurements[:, n].reshape(H.shape[0], 1)
            y = Z - (H * x)  # Innovation or Residual
            x = x + (K * y)
            # Update the error covariance
            P = (I - (K * H)) * P

        # Save states for Plotting
        xt.append(float(x[0]))ﬁ
        yt.append(float(x[1]))ﬁ
        dxt.append(float(x[2]))
        dyt.append(float(x[3]))
        ddxt.append(float(x[4]))
        ddyt.append(float(x[5]))
        Zx.append(float(Z[0]))
        Zy.append(float(Z[1]))
        Px.append(float(P[0, 0]))
        Py.append(float(P[1, 1]))
        Pdx.append(float(P[2, 2]))
        Pdy.append(float(P[3, 3]))
        Pddx.append(float(P[4, 4]))
        Pddy.append(float(P[5, 5]))
        Kx.append(float(K[0, 0]))
        Ky.append(float(K[1, 0]))
        Kdx.append(float(K[2, 0]))
        Kdy.append(float(K[3, 0]))
        Kddx.append(float(K[4, 0]))
        Kddy.append(float(K[5, 0]))

    print('Kalman Calculation done')
    return xt, yt, dxt, dyt, ddxt, ddyt, Zx, Zy, Px, Py, Pdx, Pdy, Pddx, Pddy, Kx, Ky, Kdx, Kdy, Kddx, Kddy, P


def plot_x(x):
    n = x.size  # States
    plt.scatter(float(x[0]), float(x[1]), s=100)
    plt.title('Initial Location')
    plt.show()


def plot_P(P):

    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
    plt.title('Initial Covariance Matrix $P$')
    ylocs, ylabels = plt.yticks()
    # set the locations of the yticks
    plt.yticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.yticks(np.arange(6), ('$x$', '$y$', '$\dot x$', '$\dot y$', '$\ddot x$', '$\ddot y$'), fontsize=22)

    xlocs, xlabels = plt.xticks()
    # set the locations of the yticks
    plt.xticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.xticks(np.arange(6), ('$x$', '$y$', '$\dot x$', '$\dot y$', '$\ddot x$', '$\ddot y$'), fontsize=22)

    plt.xlim([-0.5, 5.5])
    plt.ylim([5.5, -0.5])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.show()


def plot_R(R):
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(R, interpolation="none", cmap=plt.get_cmap('binary'))
    plt.title('Measurement Noise Covariance Matrix $R$')
    ylocs, ylabels = plt.yticks()
    # set the locations of the yticks
    plt.yticks(np.arange(5))
    # set the locations and labels of the yticks
    plt.yticks(np.arange(4), ('$x$', '$y$', '$\ddot x$', '$\ddot y$'), fontsize=22)

    xlocs, xlabels = plt.xticks()
    # set the locations of the yticks
    plt.xticks(np.arange(5))
    # set the locations and labels of the yticks
    plt.xticks(np.arange(4), ('$x$', '$y$', '$\ddot x$', '$\ddot y$'), fontsize=22)

    plt.xlim([-0.5, 3.5])
    plt.ylim([3.5, -0.5])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.show()


def plot_Q(Q):
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(Q, interpolation="none", cmap=plt.get_cmap('binary'))
    plt.title('Process Noise Covariance Matrix $Q$')
    ylocs, ylabels = plt.yticks()
    # set the locations of the yticks
    plt.yticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.yticks(np.arange(6), ('$x$', '$y$', '$\dot x$', '$\dot y$', '$\ddot x$', '$\ddot y$'), fontsize=22)

    xlocs, xlabels = plt.xticks()
    # set the locations of the yticks
    plt.xticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.xticks(np.arange(6), ('$x$', '$y$', '$\dot x$', '$\dot y$', '$\ddot x$', '$\ddot y$'), fontsize=22)

    plt.xlim([-0.5, 5.5])
    plt.ylim([5.5, -0.5])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.show()


def plot_measurement(m, mx, my, mpx, mpy):
    fig = plt.figure(figsize=(21, 9))
    mpx_zero = np.empty([m], dtype=float)
    mpx_zero[0:(m-1)] = 0.0
    plt.subplot(211)
    plt.step(range(m), mpx, label='$x$')
    plt.step(range(m), mpy, label='$y$')
    #plt.step(range(m), (my * 0.25), label='$a_y/4 $')
    #plt.step(range(m), mpx_zero, label='$zero$')
    plt.ylabel(r'Position + Accel $m$')
    plt.title('Measurements')
    plt.ylim([-4, 25])
    plt.legend(loc='best',prop={'size':18})

    plt.subplot(212)
    plt.step(range(m),mx, label='$a_x$')
    plt.step(range(m),my, label='$a_y$')
    plt.ylabel(r'Acceleration $m/s^2$')
    plt.ylim([-10, 10])
    plt.legend(loc='best',prop={'size':18})
    # plt.savefig('Kalman-Filter-CA-Measurements.png', dpi=72, transparent=True, bbox_inches='tight')
    plt.show()


def plot_uncertainty(Px, Py, Pddx, Pddy):
    fig = plt.figure(figsize=(16,9))
    plt.subplot(211)
    plt.plot(range(len(measurements[0])),Px, label='$x$')
    plt.plot(range(len(measurements[0])),Py, label='$y$')
    plt.title('Uncertainty (Elements from Matrix $P$)')
    plt.legend(loc='best',prop={'size':22})
    plt.subplot(212)
    plt.plot(range(len(measurements[0])),Pddx, label='$\ddot x$')
    plt.plot(range(len(measurements[0])),Pddy, label='$\ddot y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.legend(loc='best',prop={'size':22})
    plt.show()


def plot_kalman_gains(Kx, Ky, Kdx, Kdy, Kddx, Kddy):
    fig = plt.figure(figsize=(16, 9))
    plt.plot(range(len(measurements[0])), Kx, label='Kalman Gain for $x$')
    plt.plot(range(len(measurements[0])), Ky, label='Kalman Gain for $y$')
    plt.plot(range(len(measurements[0])), Kdx, label='Kalman Gain for $\dot x$')
    plt.plot(range(len(measurements[0])), Kdy, label='Kalman Gain for $\dot y$')
    plt.plot(range(len(measurements[0])), Kddx, label='Kalman Gain for $\ddot x$')
    plt.plot(range(len(measurements[0])), Kddy, label='Kalman Gain for $\ddot y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.title('Kalman Gain (the lower, the more the measurement fullfill the prediction)')
    plt.legend(loc='best', prop={'size': 18})
    plt.show()


def plot_covariance_matrix(P):
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
    plt.title('Covariance Matrix $P$ (after %i Filter Steps)' % (m))
    ylocs, ylabels = plt.yticks()
    # set the locations of the yticks
    plt.yticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.yticks(np.arange(6), ('$x$', '$y$', '$\dot x$', '$\dot y$', '$\ddot x$', '$\ddot y$'), fontsize=22)

    xlocs, xlabels = plt.xticks()
    # set the locations of the yticks
    plt.xticks(np.arange(7))
    # set the locations and labels of the yticks
    plt.xticks(np.arange(6), ('$x$', '$y$', '$\dot x$', '$\dot y$', '$\ddot x$', '$\ddot y$'), fontsize=22)

    plt.xlim([-0.5, 5.5])
    plt.ylim([5.5, -0.5])

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    # plt.savefig('Kalman-Filter-CA-CovarianceMatrix.png', dpi=72, transparent=True, bbox_inches='tight')
    plt.show()


def plot_state_vectors(xt, yt, dxt, dyt, ddxt, ddyt):
    fig = plt.figure(figsize=(16,16))

    plt.subplot(311)
    plt.step(range(len(measurements[0])),ddxt, label='$\ddot x$')
    plt.step(range(len(measurements[0])),ddyt, label='$\ddot y$')

    plt.title('Estimate (Elements from State Vector $x$)')
    plt.legend(loc='best',prop={'size':22})
    plt.ylabel(r'Acceleration $m/s^2$')
    plt.ylim([-.1,.1])

    plt.subplot(312)
    plt.step(range(len(measurements[0])),dxt, label='$\dot x$')
    plt.step(range(len(measurements[0])),dyt, label='$\dot y$')

    plt.ylabel('')
    plt.legend(loc='best',prop={'size':22})
    plt.ylabel(r'Velocity $m/s$')
    plt.ylim([-1,1])

    plt.subplot(313)
    plt.step(range(len(measurements[0])),xt, label='$x$')
    plt.step(range(len(measurements[0])),yt, label='$y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.legend(loc='best',prop={'size':22})
    plt.ylabel(r'Position $m$')
    plt.ylim([-1,1])
    # plt.savefig('Kalman-Filter-CA-StateEstimated.png', dpi=72, transparent=True, bbox_inches='tight')
    plt.show()


def plot_position(xt, yt, mpx, mpy):
    fig = plt.figure(figsize=(16, 9))
    plt.scatter(xt[0], yt[0], s=100, label='Start', c='c')
    plt.scatter(xt[-1], yt[-1], s=100, label='Goal', c='m')
    plt.plot(xt, yt, label='State Kalman', alpha=0.5, c='g')
    plt.plot(mpx, mpy, label='Measure', alpha=0.5, c='r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Position')
    plt.legend(loc='best')
    plt.xlim([-1, 22])
    plt.ylim([-4, 10])
    # plt.savefig('Kalman-Filter-CA-Position.png', dpi=72, transparent=True, bbox_inches='tight')
    plt.show()
    #plt.show()


if __name__ == '__main__':
    """ Init """
    x = init_x()
    # plot_x(x)
    P = init_P()
    # plot_P(P)
    dt = 1.0  # Time Step between Filter Steps
    A = init_A(dt)
    H = init_H()
    R = init_R()
    # plot_R(R)
    [As, Gs] = init_As_Gs()
    Q = init_Q(dt)
    # plot_Q(Q)
    I = init_I(6)
    ######
    ######
    """ Simulate """
    """
    Measurements
    Typical update rates:
    Acceleration from IMU with 10Hz
    Position from Tracking with 1Hz
    Which means, that every 10th of an acceleration measurement, there is a new position measurement from Position. 
    """
    m = 2000  # Measurements
    dx = 0.01
    # Position
    sp = 1.0  # Sigma for position
    px = 0.0  # x Position
    py = 0.0  # y Position
    [mpx_1Hz, mpy_1Hz, mpx_10Hz, mpy_10Hz, POS] = sim_measurement_pos(m, sp, px, py, dx)
    # Acceleration
    sa = 0.1  # Sigma for acceleration
    ax = 0.0  # in X
    ay = 0.0  # in Y
    [mx, my, measurements] = sim_measurement_ac(m, sa, ax, ay, mpx_1Hz, mpy_1Hz, mpx_10Hz, mpy_10Hz, dx)
    plot_measurement(m, mx, my, mpx_1Hz, mpy_1Hz)
    ######
    ######
    """ Kalman """
    [xt, yt, dxt, dyt, ddxt, ddyt, Zx, Zy, Px, Py, Pdx, Pdy, Pddx, Pddy, Kx, Ky, Kdx, Kdy, Kddx, Kddy, P] = kalman(x, P, A, H, R, Q, I, measurements, POS)
    ######
    ######
    """ Show the results """
    # plot_uncertainty(Px, Py, Pddx, Pddy)
    # plot_kalman_gains(Kx, Ky, Kdx, Kdy, Kddx, Kddy)
    # plot_covariance_matrix(P)
    # plot_state_vectors(xt, yt, dxt, dyt, ddxt, ddyt)
    plot_position(xt, yt, mpx_1Hz, mpy_1Hz)
    # dist = np.cumsum(np.sqrt(np.diff(xt) ** 2 + np.diff(yt) ** 2))
    # print('Your drifted %d units from origin.' % dist[-1])



