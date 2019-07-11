"""

Inverted Pendulum MPC control

author: Atsushi Sakai

"""

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import cvxpy
import ecos
import cvxopt
from cvxopt import matrix
import scipy.linalg
import scipy.sparse as sp


l_bar = 2.0  # length of bar
M = 10  # [kg]
m = 1  # [kg]
g = 9.8  # [m/s^2]

nx = 4   # number of state
nu = 1   # number of input
T = 15  # Horizon length
N = T
Q = np.eye(nx)
R = np.eye(nu)*0.0
P = np.eye(nx)*0.1
delta_t = 0.1  # time tick

animation = True

x_start = -2
x_end = 1
def main():
    x0 = np.array([
        [x_start-x_end],     # start x position
        [0.0],
        [0.0],     # start theta
        [0.0]
    ])

    x = np.copy(x0)
    cnt = 70

    for i in range(cnt):
        start = time.time()
        ox, dx, otheta, dtheta, ou = opt_mpc_control(x)
        print("calc time:{0} [sec]".format(time.time() - start))
        u = ou[0]
        x = A @ x + B @ [[u]]

        if animation:
            plt.clf()
            px = float(x[0]+x_end)
            theta = float(x[2])
            show_cart(px, theta)
            plt.xlim([-4.0, 4.0])
            plt.ylim([-0.01, 4.0])
            plt.pause(0.0001)


def generate_inequalities_constraints_mat(N, nx, nu, xmin, xmax, umin, umax):
    """
    generate matrices of inequalities constrints

    return G, h
    """
    G = np.zeros((0, (nx + nu) * N))
    h = np.zeros((0, 1))
    if umax is not None:
        tG = np.hstack([np.eye(N * nu), np.zeros((N * nu, nx * N))])
        th = np.kron(np.ones((N * nu, 1)), umax)
        G = np.vstack([G, tG])
        h = np.vstack([h, th])

    if umin is not None:
        tG = np.hstack([np.eye(N * nu) * -1.0, np.zeros((N * nu, nx * N))])
        th = np.kron(np.ones((N, 1)), umin * -1.0)
        G = np.vstack([G, tG])
        h = np.vstack([h, th])

    if xmax is not None:
        tG = np.hstack([np.zeros((N * nx, nu * N)), np.eye(N * nx)])
        th = np.kron(np.ones((N, 1)), xmax)
        G = np.vstack([G, tG])
        h = np.vstack([h, th])

    if xmin is not None:
        tG = np.hstack([np.zeros((N * nx, nu * N)), np.eye(N * nx) * -1.0])
        th = np.kron(np.ones((N, 1)), xmin * -1.0)
        G = np.vstack([G, tG])
        h = np.vstack([h, th])

    return G, h

def opt_mpc_with_state_constr(A, B, N, Q, R, P, x0, xmin=None, xmax=None, umax=None, umin=None):
    """
    optimize MPC problem with state and (or) input constraints

    return
        x: state
        u: input
    """
    (nx, nu) = B.shape

    H = scipy.linalg.block_diag(np.kron(np.eye(N), R), np.kron(
        np.eye(N - 1), Q), np.eye(P.shape[0]))
    #  print(H)

    # calc Ae
    Aeu = np.kron(np.eye(N), -B)
    #  print(Aeu)
    #  print(Aeu.shape)
    Aex = scipy.linalg.block_diag(np.eye((N - 1) * nx), P)
    Aex -= np.kron(np.diag([1.0] * (N - 1), k=-1), A)
    #  print(Aex)
    #  print(Aex.shape)
    Ae = np.hstack((Aeu, Aex))
    #  print(Ae.shape)

    # calc be
    be = np.vstack((A, np.zeros(((N - 1) * nx, nx)))) @ x0
    #  print(be)

    np.set_printoptions(precision=3)
    #  print(H.shape)
    #  print(H)
    #  print(np.zeros((N * nx + N * nu, 1)))
    #  print(Ae)
    #  print(be)

    # === optimization ===
    P = matrix(H)
    q = matrix(np.zeros((N * nx + N * nu, 1)))
    A = matrix(Ae)
    b = matrix(be)

    if umax is None and umin is None:
        sol = cvxopt.solvers.qp(P, q, A=A, b=b)
    else:
        G, h = generate_inequalities_constraints_mat(
            N, nx, nu, xmin, xmax, umin, umax)
        #  print(G)
        #  print(h)

        G = matrix(G)
        h = matrix(h)

        sol = cvxopt.solvers.qp(P, q, G, h, A=A, b=b)

    #  print(sol)
    fx = np.array(sol["x"])
    #  print(fx)

    u = fx[0:N * nu].reshape(N, nu).T

    x = fx[-N * nx:].reshape(N, nx).T
    x = np.hstack((x0, x))
    #  print(x)
    #  print(u)

    return x, u

def opt_mpc_control(x0):
    N = 10
    # x, u = opt_mpc_with_state_constr(A, B, N, Q, R, P, x0)
    x, u = ecos_mpc_with_state_constr(A, B, N, Q, R, P, x0)
    ox = np.array(x[:, 0]).flatten()
    dx = np.array(x[:, 1]).flatten()
    theta = np.array(x[:, 2]).flatten()
    dtheta = np.array(x[:, 3]).flatten()
    ou = np.array(u).flatten()
    return ox, dx, theta, dtheta, ou


def use_modeling_tool(A, B, N, Q, R, P, x0, umax=None, umin=None, xmin=None, xmax=None):
    """
    solve MPC with modeling tool for test
    """
    (nx, nu) = B.shape

    # mpc calculation
    x = cvxpy.Variable((nx, N + 1))
    u = cvxpy.Variable((nu, N))

    costlist = 0.0
    constrlist = []

    for t in range(N):
        costlist += 0.5 * cvxpy.quad_form(x[:, t], Q)
        costlist += 0.5 * cvxpy.quad_form(u[:, t], R)

        constrlist += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]

        if xmin is not None:
            constrlist += [x[:, t] >= xmin[:, 0]]
        if xmax is not None:
            constrlist += [x[:, t] <= xmax[:, 0]]

    costlist += 0.5 * cvxpy.quad_form(x[:, N], P)  # terminal cost
    if xmin is not None:
        constrlist += [x[:, N] >= xmin[:, 0]]
    if xmax is not None:
        constrlist += [x[:, N] <= xmax[:, 0]]

    if umax is not None:
        constrlist += [u <= umax]  # input constraints
    if umin is not None:
        constrlist += [u >= umin]  # input constraints

    constrlist += [x[:, 0] == x0[:, 0]]  # inital state constraints

    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)

    prob.solve(verbose=False)

    return x.value, u.value

def mpc_control(x0):
    N = 30
    x, u = use_modeling_tool(A, B, N, Q, R, P, x0)
    ox = np.array(x[0, :]).flatten()
    dx = np.array(x[1, :]).flatten()
    theta = np.array(x[2, :]).flatten()
    dtheta = np.array(x[3, :]).flatten()
    ou = np.array(u).flatten()
    return ox, dx, theta, dtheta, ou


def ecosqp(H, f, A=None, B=None, Aeq=None, Beq=None):
    """
    solve a quadratic programing problem with ECOS

        min 1/2*x'*H*x + f'*x
        s.t. A*x <= b
             Aeq*x = beq

    return sol
        It is same data format of CVXOPT.

    """
    # ===dimension and argument checking===
    # H
    assert H.shape[0] == H.shape[1], "Hessian must be a square matrix"

    n = H.shape[0]

    # f
    if (f is None) or (f.size == 0):
        f = np.zeros((n, 1))
    else:
        assert f.shape[0] == n, "Linear term f must be a column vector of length"
        assert f.shape[1] == 1, "Linear term f must be a column vector"

    # check cholesky
    try:
        W = np.linalg.cholesky(H).T
    except np.linalg.linalg.LinAlgError:
        W = scipy.linalg.sqrtm(H)
    #  print(W)

    # set up SOCP problem
    c = np.vstack((np.zeros((n, 1)), 1.0))
    #  print(c)

    # pad Aeq with a zero column for t
    if Aeq is not None:
        Aeq = np.hstack((Aeq, np.zeros((Aeq.shape[0], 1))))
        beq = Beq
    else:
        Aeq = np.matrix([])
        beq = np.matrix([])

    # create second-order cone constraint for objective function
    fhalf = f / math.sqrt(2.0)
    #  print(fhalf)
    zerocolumn = np.zeros((W.shape[1], 1))
    #  print(zerocolumn)

    tmp = 1.0 / math.sqrt(2.0)

    Gquad1 = np.hstack((fhalf.T, np.matrix(-tmp)))
    Gquad2 = np.hstack((-W, zerocolumn))
    Gquad3 = np.hstack((-fhalf.T, np.matrix(tmp)))
    Gquad = np.vstack((Gquad1, Gquad2, Gquad3))
    #  print(Gquad1)
    #  print(Gquad2)
    #  print(Gquad3)
    #  print(Gquad)

    hquad = np.vstack((tmp, zerocolumn, tmp))
    #  print(hquad)

    if A is None:
        G = Gquad
        h = hquad
        dims = {'q': [W.shape[1] + 2], 'l': 0}
    else:
        G1 = np.hstack((A, np.zeros((A.shape[0], 1))))
        G = np.vstack((G1, Gquad))
        h = np.vstack((B, hquad))
        dims = {'q': [W.shape[1] + 2], 'l': A.shape[0]}

    c = np.array(c).flatten()
    G = sp.csc_matrix(G)
    h = np.array(h).flatten()

    if Aeq.size == 0:
        sol = ecos.solve(c, G, h, dims, verbose=False)
    else:
        Aeq = sp.csc_matrix(Aeq)
        beq = np.array(beq).flatten()
        sol = ecos.solve(c, G, h, dims, Aeq, beq, verbose=False)
    #  print(sol)
    #  print(sol["x"])

    sol["fullx"] = sol["x"]
    sol["x"] = sol["fullx"][:n]
    sol["fval"] = sol["fullx"][-1]

    return sol

def ecos_mpc_with_state_constr(A, B, N, Q, R, P, x0, xmin=None, xmax=None, umax=None, umin=None):
    """
    optimize MPC problem with state and (or) input constraints

    return
        x: state
        u: input
    """
    (nx, nu) = B.shape

    H = scipy.linalg.block_diag(np.kron(np.eye(N), R), np.kron(
        np.eye(N - 1), Q), np.eye(P.shape[0]))

    # calc Ae
    Aeu = np.kron(np.eye(N), -B)
    #  print(Aeu)
    #  print(Aeu.shape)
    Aex = scipy.linalg.block_diag(np.eye((N - 1) * nx), P)
    #  print(Aex)
    Aex -= np.kron(np.diag([1.0] * (N - 1), k=-1), A)
    #  print(np.diag([1.0] * (N - 1), k=-1))

    Ae = np.hstack((Aeu, Aex))
    #  print(Ae)


    # calc be
    be = np.vstack((A, np.zeros(((N - 1) * nx, nx)))) @ x0
    #  print(be)
    #  print(be.shape)

    # === optimization ===
    q = np.zeros((N * nx + N * nu, 1))

    if umax is None and umin is None:
        sol = ecosqp(H, q, Aeq=Ae, Beq=be)
    else:
        G, h = generate_inequalities_constraints_mat(
            N, nx, nu, xmin, xmax, umin, umax)

        # print(h)
        # print(G)

        sol = ecosqp(H, q, A=G, B=h, Aeq=Ae, Beq=be)

    #  print(sol)
    fx = np.array(sol["x"])

    u = fx[0:N * nu].reshape(N, nu).T
    x = fx[-N * nx:].reshape(N, nx).T
    x = np.hstack((x0, x))
    #  print(x)
    #  print(u)

    return x, u


def get_model_matrix():

    # Model Parameter
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]
    ])
    A = np.eye(nx) + delta_t * A

    B = np.array([
        [0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]
    ])
    B = delta_t * B

    return A, B

A, B = get_model_matrix()

def flatten(a):
    return np.array(a).flatten()

def show_cart(xt, theta):
    cart_w = 0.8
    cart_h = 0.4
    radius = 0.1

    cx = np.matrix([-cart_w / 2.0, cart_w / 2.0, cart_w /
                    2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.matrix([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.matrix([0.0, l_bar * math.sin(-theta)])
    bx += xt
    by = np.matrix([cart_h, l_bar * math.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(5.0))
    ox = [radius * math.cos(a) for a in angles]
    oy = [radius * math.sin(a) for a in angles]

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + float(bx[0, -1])
    wy = np.copy(oy) + float(by[0, -1])

    plt.plot(flatten(cx), flatten(cy), "-b")
    plt.plot(flatten(bx), flatten(by), "-y")
    plt.plot(flatten(rwx), flatten(rwy), "-k")
    plt.plot(flatten(lwx), flatten(lwy), "-k")
    plt.plot(flatten(wx), flatten(wy), "-",color='orange')
    plt.title("x:" + str(round(xt, 5)) + ",theta:" +
              str(round(math.degrees(theta), 5)))

    plt.axis("equal")
    plt.autoscale_on = False


if __name__ == '__main__':
    main()
    #  visualize_test()
