"""
This module implements Kalman filters
"""
import numpy as np

####################
# Helper functions #
####################


def __kalman_forward_update__(xhat_fm, P_fm, y, u, A, B, C, R, Q):
    """
    :param xhat_fm:
    :param P_fm:
    :param y:
    :param u:
    :param A:
    :param B:
    :param C:
    :param R:
    :param Q:
    :return:
    """
    I = np.array(np.eye(A.shape[0]))
    gammaW = np.array(np.eye(A.shape[0]))

    K_f = P_fm@C.T@np.linalg.pinv(C@P_fm@C.T + R)

    if y is not None and not np.isnan(y).any():
        xhat_fp = xhat_fm + K_f@(y - C@xhat_fm)
        P_fp = (I - K_f@C)@P_fm
        xhat_fm = A@xhat_fp + B@u
        P_fm = A@P_fp@A.T + gammaW@Q@gammaW.T
    else:
        xhat_fp = xhat_fm
        P_fp = (I - K_f@C)@P_fm
        xhat_fm = A@xhat_fp + B@u
        P_fm = A@P_fp@A.T + gammaW@Q@gammaW.T
        
    return xhat_fp, xhat_fm, P_fp, P_fm


def __kalman_forward_filter__(xhat_fm, P_fm, y, u, A, B, C, R, Q):
    """
    :param xhat_fm:
    :param P_fm:
    :param y:
    :param u:
    :param A:
    :param B:
    :param C:
    :param R:
    :param Q:
    :return:
    """
    if u is None:
        u = np.array(np.zeros([B.shape[1], y.shape[1]]))

    xhat_fp = None
    P_fp = []
    P_fm = [P_fm]

    for i in range(y.shape[1]):
        _xhat_fp, _xhat_fm, _P_fp, _P_fm = __kalman_forward_update__(xhat_fm[:, [-1]], P_fm[-1], y[:, [i]], u[:, [i]],
                                                                     A, B, C, R, Q)
        if xhat_fp is None:
            xhat_fp = _xhat_fp
        else:
            xhat_fp = np.hstack((xhat_fp, _xhat_fp))
        xhat_fm = np.hstack((xhat_fm, _xhat_fm))

        P_fp.append(_P_fp)
        P_fm.append(_P_fm)

    return xhat_fp, xhat_fm, P_fp, P_fm


def __kalman_backward_smooth__(xhat_fp, xhat_fm, P_fp, P_fm, A):
    """
    :param xhat_fp:
    :param xhat_fm:
    :param P_fp:
    :param P_fm:
    :param A:
    :return:
    """
    N = xhat_fp.shape[1]

    xhat_smooth = copy.copy(xhat_fp)
    P_smooth = copy.copy(P_fp)
    for t in range(N-2, -1, -1):
        L = P_fp[t]@A.T@np.linalg.pinv(P_fm[t])
        xhat_smooth[:, [t]] = xhat_fp[:, [t]] + L@(xhat_smooth[:, [t+1]] - xhat_fm[:, [t+1]])
        P_smooth[t] = P_fp[t] - L@(P_smooth[t+1] - P_fm[t+1])

    return xhat_smooth, P_smooth


def known_dynamics(x, params, u=None, options=None):
    """
    Run a forward RTS Kalman smoother given known dynamics to estimate the derivative.

    :param x: matrix of time series of (noisy) measurements
    :type x: np.array (float)

    :param params: a list of:
                    - x0: inital condition, matrix of Nx1, N = number of states
                    - P0: initial covariance matrix of NxN
                    - A: dynamics matrix, NxN
                    - B: control input matrix, NxM, M = number of measurements
                    - C: measurement dynamics, MxN
                    - R: covariance matrix for the measurements, MxM
                    - Q: covariance matrix for the model, NxN
    :type params: list (matrix)

    :param u: matrix of time series of control inputs
    :type u: np.array (float)

    :param options: a dictionary indicating whether to run smoother
    :type params: dict {'smooth': boolean}, optional

    :return: matrix:
            - xhat_smooth: smoothed estimates of the full state x

    :rtype: tuple -> (np.array, np.array)
    """
    if len(x.shape) == 2:
        y = x
    else:
        y = np.reshape(x, [1, len(x)])

    if options is None:
        options = {'smooth': True}

    x0, P0, A, B, C, R, Q = params

    xhat_fp, xhat_fm, P_fp, P_fm = __kalman_forward_filter__(x0, P0, y, u, A, B, C, R, Q)
    xhat_smooth, _ = __kalman_backward_smooth__(xhat_fp, xhat_fm, P_fp, P_fm, A)

    if not options['smooth']:
        return xhat_fp

    return xhat_smooth


