import numpy as np
from scipy.optimize import minimize, LinearConstraint


def MinMpc2_WithoutTC(L, gamma_prev, gamma_dot_prev, step, N):
    """
    MPC without Terminal Cost for UAV coordination.
    Inputs:
        L: Laplacian matrix (n x n)
        gamma_prev: previous virtual time (n x N)
        gamma_dot_prev: previous gamma dot (n x N)
        step: time step
        N: prediction horizon
    Outputs:
        gamma: virtual time trajectory (n x N)
        gamma_dot: derivative of virtual time (n x N)
        u: control inputs (n x N)
    """
    n = gamma_prev.shape[0]
    m = 1
    x0 = np.vstack((gamma_prev[:, 0], gamma_dot_prev[:, 0])) # Flatten in column-major
    x0 = x0.flatten(order='F').reshape(-1, 1)

    A = np.array([[1, step],
                  [0, 1]])
    B = np.array([[0.5 * step ** 2],
                  [step]])

    gamma = np.zeros((n, N))
    gamma_dot = np.zeros((n, N))
    u = np.zeros((n, N))

    for ll in range(n):
        Q = np.array([[L[ll, ll], 0],
                      [0, 1]])
        R = np.eye(m)

        Fx = np.array([[1, 0],
                       [0, 1],
                       [-1, 0],
                       [0, -1]])
        gx = np.array([[1000], [10], [0], [0]])

        Fu = np.array([[1],
                       [-1]])
        gu = np.array([100, 100])

        nn = 2
        AX = np.zeros((nn * (N + 1), nn))
        BU = np.zeros((nn * (N + 1), m * N))

        for i in range(N + 1):
            AX[i * nn:(i + 1) * nn, :] = np.linalg.matrix_power(A, i)
            for j in range(N):
                if i > j:
                    BU[i * nn:(i + 1) * nn, j * m:(j + 1) * m] = np.linalg.matrix_power(A, i - j - 1) @ B
        # print(AX)
        QX = Q.copy()
        RU = R.copy()
        FX = Fx.copy()
        gX = gx.copy()
        FU_blk = Fu.copy()
        gU = gu.copy()

        for i in range(N - 1):
            QX = np.block([[QX, np.zeros((QX.shape[0], Q.shape[1]))],
                           [np.zeros((Q.shape[0], QX.shape[1])), Q]])
            RU = np.block([[RU, np.zeros((RU.shape[0], R.shape[1]))],
                           [np.zeros((R.shape[0], RU.shape[1])), R]])
            FX = np.block([[FX, np.zeros((FX.shape[0], Fx.shape[1]))],
                           [np.zeros((Fx.shape[0], FX.shape[1])), Fx]])
            gX = np.concatenate((gX, gx), axis=0)
            FU_blk = np.block([[FU_blk, np.zeros((FU_blk.shape[0], Fu.shape[1]))],
                               [np.zeros((Fu.shape[0], FU_blk.shape[1])), Fu]])
            gU = np.concatenate((gU, gu), axis=0)

        QX = np.block([[QX, np.zeros((QX.shape[0], Q.shape[1]))],
                       [np.zeros((Q.shape[0], QX.shape[1])), Q]])
        FX = np.block([[FX, np.zeros((FX.shape[0], Fx.shape[1]))],
                       [np.zeros((Fx.shape[0], FX.shape[1])), Fx]])
        gX = np.concatenate((gX, gx), axis=0)
        # print(gX)

        H = BU.T @ QX @ BU + RU

        row = -L[ll, :].copy().reshape(1, -1)  # shape (1, n)
        row[0, ll] = 0                         # zero diagonal
        row = row @ gamma_prev             # shape (1, N)
        # print(row)

        # if N > 1:

        #     v = -np.concatenate(([2 * row] * N + [2 * row + 2 * (row - row)]))  # dummy 2nd derivative approx
        # else:
        #     v = -np.concatenate(([2 * row] * N + [2 * row + 2 * step]))
        if N > 1:
            top_row = -np.append(2 * row[0,:], 2 * row[0,-1] + 2 * (row[0,-1] - row[0,-2]))  # last element
            bottom_row =-2 *np.ones(N+1)
        else:
            top_row = -np.append(2 * row[0,:], 2 * row[0,-1] + 2 * step)
            bottom_row = -2*np.ones(1,N+1)
        
        v = np.array([top_row,bottom_row])
        v = v.T
        v = v.reshape(-1,1)
        v=v.T
       
        xk = x0[2 * ll:2 * ll + 2]
        l = v @ BU
        
        qk = 2 * xk.T @ AX.T @ QX @ BU
        def cost(z):
            return z.T @ H @ z + qk @ z + l @ z

        F = np.vstack((FX @ BU, FU_blk))
        aa = gX - FX @ AX @ xk
        g = np.append(aa[:,0],gU)
        # print(g)
        
        z0 = np.zeros(N)
        
        lower_bounds = -np.inf * np.ones_like(g)
        linear_constraint = LinearConstraint(F, lower_bounds, g)

        res = minimize(cost, z0, method='trust-constr', 
                       constraints=[linear_constraint],
                       options={
                            'xtol': 1e-10,        # Tolerance for solution vector x
                            'gtol': 1e-10,        # Tolerance for gradient norm
                            'barrier_tol': 1e-10, # Tolerance for interior-point barrier
                            'maxiter': 1000,      # Increase max number of iterations
                            'verbose': 1,
                            'disp': False         # Show optimization progress
               })

                                    

        z_opt = res.x
        u[ll, :] = z_opt
        z_opt = np.array([z_opt])

        bb = AX @ xk + BU @ z_opt.T
        # print(bb)
        gamma[ll, :] = bb[0::2,0][1:]  # skip initial state
        gamma_dot[ll, :] = bb[1::2,0][1:]

    return gamma, gamma_dot, u



L=np.array([[1, -1],[-1, 1]])
gamma_prev =  np.array([[0, 0.1,0.2],[2, 2.1,2.2]])      
gamma_dot_prev=np.ones((2,2))     
gamma, gamma_dot, u= MinMpc2_WithoutTC(L, gamma_prev, gamma_dot_prev, 0.1, 3)
print(gamma)
print(gamma_dot)
print(u)