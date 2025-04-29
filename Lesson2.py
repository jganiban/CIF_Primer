# ------------------------------------------------------------------
# Lesson2.py
# This module implements an example of computing a control invariant
# ellipsoidal set for a discrete-time linear system using cvxpy.
#
# Justin Ganiban
# 4/23/25
# ------------------------------------------------------------------

# Section 1: Defining the problem.
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Define LTI system.
A = np.array([[0, 1],[0, 0]])
An, Am = A.shape
B = np.array([[0],[1]])
Bn, Bm = B.shape
E = np.array([[0],[1]])
En, Em = E.shape

# Define velocity and input constraints - |v| <= vmax, |u| <= umax.
vmax = 20.0
v_con = np.array([[0, 1],[0, -1]])
v_bounds = np.array([[vmax, -vmax]])

# Drag uncertainty coefficients
cd = 1.4
L = cd*vmax

umax = 10.0
u_con = np.array([[1],[-1]])
u_bounds = np.array([[umax], [-umax]])

# Define Uncertainty model: q = Cx+Du, norm(p)<=cd*vmax*norm(q)
C = np.array([[0, 1]])

# Define the decision variables.
Q = cp.Variable((An, An),symmetric=True)
Y = cp.Variable((Bm, Bn))
nu = cp.Variable()

# Define the decay rate that bounds the Lyapunov function.
alpha = 0.1

# ------------------------------------------------------------------
# Section 2: Set up constraints for cvxpy.
constraints = []

# Define the Lyapunov constraint.
lyap_con = cp.bmat([
    [A@Q+B@Y+Q@A.T+Y.T@B.T+alpha*Q, (nu)*E,                (C@Q).T],
    [(nu)*E.T,                      np.array([[-nu]]),   np.array([[0]])],
    [C@Q,                           np.array([[0]]),       np.array([[-nu/L**2]])]
])
constraints += [lyap_con << 0]
constraints += [nu > 0]

# Ensure a positive definite Q.
constraints += [Q >> 1e-4 * np.eye(An)]

# Set up LMI constraint on velocity.
for ii in range(len(v_bounds)):
    a_i = v_con[ii,:].reshape((1,An))
    b_i = cp.Constant([[v_bounds[ii,0]**2]])
    vLMI = cp.bmat([
            [b_i, a_i@Q],
            [Q@a_i.T, Q]
    ])
    constraints += [vLMI >> 0]

# Set up LMI constraint on input.
for jj in range(len(u_bounds)):
    c_i = u_con[jj, :].reshape((1, Bm))        
    Yci = c_i @ Y                         
    d_i = cp.Constant([[u_bounds[jj,0]**2]])         

    uLMI = cp.bmat([
        [d_i,     Yci],    
        [Yci.T,    Q]         
    ])
    constraints += [uLMI >> 0]

# Define Objective function: Maximize ellipsoid by max(logdet(Q))
objective = cp.Maximize(cp.log_det(Q))

# ------------------------------------------------------------------
# Section 3: Solve the problem.
prob = cp.Problem(objective, constraints)

print("----------------------------------")
prob.solve(solver=cp.CLARABEL)
print("solver status : " + prob.status)
print("solve time    :" + str(prob.solver_stats.solve_time))
# print("cost          :" + str(prob.objective.value))
print("----------------------------------")

# Back out the optimal control gain K.
K = Y.value @ np.linalg.inv(Q.value)