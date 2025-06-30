# ------------------------------------------------------------------
# Lesson5.py
# This module implements a control invariant set for equilibrium/hover
# control for 6-DOF rigid body rocket dynamics
#
# Justin Ganiban
# 6/12/25
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Section 1: Defining the dynamics.
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import simplify, cos, sin, atan2, asin, exp, tan
from scipy.linalg import block_diag

# Helper function for cross products and euler angle/pqr derivatives
def skew(v):
    return Matrix([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

def get_R_from_euler(e) :
    phi = e[0]
    theta = e[1]
    psi = e[2]

    C_B_I = zeros(3,3)
    C_B_I[0,0] = cos(psi) * cos(theta)
    C_B_I[0,1] = sin(psi) * cos(theta)
    C_B_I[0,2] = -sin(theta)
    C_B_I[1,0] = -sin(psi)*cos(phi) + cos(psi)*sin(theta)*sin(phi)
    C_B_I[1,1] = cos(psi)*cos(phi) + sin(psi)*sin(theta)*sin(phi)
    C_B_I[1,2] = cos(theta)*sin(phi)
    C_B_I[2,0] = sin(psi)*sin(phi) + cos(psi)*sin(theta)*cos(phi)
    C_B_I[2,1] = -cos(psi)*sin(phi)+sin(psi)*sin(theta)*cos(phi)
    C_B_I[2,2] = cos(theta)*cos(phi)

    return C_B_I

def get_T(e) :
    phi = e[0]
    theta = e[1]
    # psi = e[2]

    T = zeros(3,3)
    T[0,0] = 1
    T[0,1] = sin(phi) * tan(theta)
    T[0,2] = cos(phi) * tan(theta)
    T[1,0] = 0
    T[1,1] = cos(phi)
    T[1,2] = -sin(phi)
    T[2,0] = 0
    T[2,1] = sin(phi) / cos(theta)
    T[2,2] = cos(phi) / cos(theta)
    return T

# Defining system parameters.
m = 1
J_B = 0.01 * np.eye(3) # 1st MOI
Jx, Jy, Jz = np.diag(J_B)

r_t = 0.01 # gimbal point unit vector
g = 1 # unit gravity

x = Matrix(symbols('rx ry rz vx vy vz phi theta psi p q r', real=True))
F = Matrix(symbols('Fx Fy Fz',real=True))
tau = Matrix(symbols('Tx Ty Tz',real=True))
u = Matrix([F,tau])

C_BI = get_R_from_euler(x[6:9])
C_IB = C_BI.transpose()
Tmatrix = get_T(x[6:9])

g_I = g*Matrix(np.array([0, 0, -1]))
r_T_b = r_t*Matrix(np.array([0, 0, -1]))

# Define dynamics f(x,u)
f = zeros(12,1)
f[0:3, 0] = x[3:6, 0]
f[3:6, 0] = 1/m * C_IB*F + g_I
f[6:9, 0] = Tmatrix * x[9:, 0]
f[9:, 0] = np.linalg.inv(J_B) * (tau+skew(r_T_b)*F - skew(x[9:12, 0]) * J_B * x[9:12, 0])


# Define new state eta and control xi for equilibrium control
eta = x.copy()
u_eq = Matrix(np.array([0, 0, m*g, 0, 0, 0]))

x_eq = {xi: 0 for xi in x}
u_eq_dict = {ui: val for ui, val in zip(u, u_eq)}

# Linearize system wrt equilibrium point.
A_sym = f.jacobian(x)
B_sym = f.jacobian(u)

subs = {**x_eq, **u_eq_dict}

A_eval = A_sym.evalf(subs=subs)
B_eval = B_sym.evalf(subs=subs)

A = np.array(A_eval).astype(np.float64)
B = np.array(B_eval).astype(np.float64)
An, Am = A.shape
Bn, Bm = B.shape

# Define input constraint.
umax = 7.0
umin = -7.0

# Define constraint directions
u_con = np.vstack([
    np.eye(Bm),       # u_i ≤ umax
    -np.eye(Bm)       # -u_i ≤ -umin,  u_i ≥ umin
])  # shape (2*Bm, Bm)

u_bounds = np.vstack([
    umax * np.ones((Bm, 1)),    # upper bounds
    -umin * np.ones((Bm, 1))    # lower bounds 
]) 

# Define uncertainty dynamics E*p
E = np.vstack((zeros(3,9),eye(9)))

# p = phi(q), q = Cx+Du. Defining C and D here
C1 = np.hstack((zeros(6,6),np.vstack((eye(3),zeros(3,3))), zeros(6,3)))
D1 = np.hstack((np.vstack((zeros(3,3),eye(3))),zeros(6,3)))
C2 = np.hstack((zeros(4,6),np.array([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]])))
D2 = zeros(4,6)
C3 = np.hstack((zeros(3,9),eye(3)))
D3 = zeros(3,6)

C = np.vstack((C1,C2,C3))
D = np.vstack((D1,D2,D3))

# Define the decision variables.
Q = cp.Variable((An, An),symmetric=True)
Y = cp.Variable((Bm, Bn))
lambda_ = cp.Variable(3, pos=True)

# Define N1 - block diagonal LMI for q
zeros_64 = np.zeros((6, 4))
zeros_63 = np.zeros((6, 3))
zeros_46 = np.zeros((4, 6))
zeros_43 = np.zeros((4, 3))
zeros_36 = np.zeros((3, 6))
zeros_34 = np.zeros((3, 4))
N1_1 = lambda_[0]*np.eye(6)
N1_2 = lambda_[1]*np.eye(4)
N1_3 = lambda_[2]*np.eye(3)
row1 = cp.hstack([N1_1, zeros_64, zeros_63])
row2 = cp.hstack([zeros_46, N1_2, zeros_43])
row3 = cp.hstack([zeros_36, zeros_34, N1_3])
N1 = cp.vstack([row1, row2, row3])

# Define N2 - block diagonal LMI for p
gamma = np.array([0.2, 0.2, 0.2])

N2_1 = (gamma[0]**2 * lambda_[0]) * cp.Constant(np.eye(3))
N2_2 = (gamma[1]**2 * lambda_[1]) * cp.Constant(np.eye(3))
N2_3 = (gamma[2]**2 * lambda_[2]) * cp.Constant(np.eye(3))

# Assemble block diagonal with cp.bmat — all sizes match (3×3 blocks)
N2 = cp.bmat([
    [N2_1, 0*np.eye(3), 0*np.eye(3)],
    [0*np.eye(3), N2_2, 0*np.eye(3)],
    [0*np.eye(3), 0*np.eye(3), N2_3]
])

# Define the decay rate that bounds the Lyapunov function.
alpha = 0.2

# ------------------------------------------------------------------
# Section 2: Set up constraints for cvxpy.
constraints = []

# Define the Lyapunov LMI constraint.
lyap_con = cp.bmat([
    [A@Q + B@Y + Q@A.T + Y.T@B.T + alpha*Q,   E@N2.T,                             (C@Q+D@Y).T],
    [N2@E.T,                                  -N2,              cp.Constant(np.zeros((9,13)))],
    [C@Q+D@Y,                       cp.Constant(np.zeros((13,9))),                        -N1]
])
constraints += [lyap_con << 0]

# Ensure positive lambda
constraints += [lambda_[i] >= 1e-4 for i in range(3)]

# Ensure a positive definite Q.
constraints += [Q >> 1e-6 * np.eye(An)]

# Set up LMI constraint on input. Make sure to account for feedforward force
u_eq_vec = np.array(u_eq).flatten()
for jj in range(len(u_bounds)):
    c_i = u_con[jj, :].reshape((1, Bm))  
    eq_bias = c_i @ u_eq_vec  
    d_i_val = u_bounds[jj, 0]**2 - float(eq_bias @ eq_bias.T)

    Yci = c_i @ Y                         
    d_i = cp.Constant([[u_bounds[jj,0]**2]])         

    uLMI = cp.bmat([
        [d_i,     Yci],    
        [Yci.T,    Q]         
    ])
    constraints += [uLMI >> 0]

# Define Objective function: Maximize ellipsoid by max(logdet(Q))
objective = cp.Maximize(cp.log_det(Q))

# objective = cp.Minimize(cp.tr_inv(Q))

# ------------------------------------------------------------------
# Section 3: Solve the problem.
prob = cp.Problem(objective, constraints)

print("----------------------------------")
prob.solve(solver=cp.MOSEK, verbose=True)
print("solver status : " + prob.status)
print("solve time    :" + str(prob.solver_stats.solve_time))
print("cost          :" + str(prob.objective.value))
print("----------------------------------")

# Back out the optimal control gain K.
K = Y.value @ np.linalg.inv(Q.value)
# print(K)

# ------------------------------------------------------------------
# Section 4: Extracting the Ellipse and plotting

# Extract the eigenvalues and eigenvectors of Q.
Q_val = Q.value
eigvals, eigvecs = np.linalg.eigh(Q_val)

# Use the two dominant directions for plotting
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Create ellipse points in unit circle
theta = np.linspace(0, 2*np.pi, 100)
circle = np.vstack((np.cos(theta), np.sin(theta)))  

# Scale and rotate by Q eigenstructure (for 2D subspace)
scale = np.sqrt(eigvals[:2])
ellipse_points = eigvecs[:, :2] @ (scale[:, None] * circle)  

# Plot rx, ry
plt.figure()
plt.plot(ellipse_points[0, :], ellipse_points[1, :])
plt.gca().set_aspect('equal')
plt.title("Ellipsoidal Level Set from Q")
plt.xlabel("rx")
plt.ylabel("ry")
plt.grid(True)
plt.show()


# ------------------------------------------------------------------
# Section 5: Simulate dynamics with IC and optimal K

from scipy.integrate import solve_ivp
import random

def closed_loop_dynamics(t, x, A, B, K, u_eq):
    u_eq_vec = np.array(u_eq).flatten()
    xi = u_eq_vec + K@x
    xdot = A@x + B@xi
    return xdot

# Sample initial conditions evenly from the ellipsoid boundary
initial_conditions = []
num_traj = 25
for ii in range(num_traj):
    x0x = random.choice(ellipse_points[0,:])
    x0_idx = np.where(ellipse_points[0,:] == x0x)[0]
    x0y = ellipse_points[1, x0_idx]
    x0 = np.array([x0x, x0y[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    initial_conditions.append(x0)

t_span = (0, 50)
t_vec = np.linspace(*t_span, 100)

traj = []
u_traj = np.empty((len(initial_conditions),len(t_vec)))
for ii in range(len(initial_conditions)):
    traj.append(solve_ivp(closed_loop_dynamics, t_span, initial_conditions[ii],'RK45',t_eval=t_vec, args=(A, B, K, u_eq)))

# Overlay the trajectory in state space with the ellipsoid
plt.figure(figsize=(6, 6))
plt.plot(ellipse_points[0,:], ellipse_points[1,:], label="Ellipsoid")
# plt.plot(ellipse_points[:, 0], ellipse_points[:, 1], label="0.5 Level Set")
for ii in range(len(initial_conditions)):
    plt.plot(traj[ii].y[0], traj[ii].y[1])
plt.xlabel("rx")
plt.ylabel("ry")
plt.title("Trajectory within Control Invariant Ellipsoid")
plt.grid(True)
plt.legend()
plt.xlim([-np.max(ellipse_points[0,:])-10, np.max(ellipse_points[0,:])+10])
plt.ylim([-np.max(ellipse_points[1,:])-10, np.max(ellipse_points[1,:])+10])

plt.show()

# Plot input u vs t
# for ii in range(2*num_traj):
#     for jj in range(len(t_vec)):
#         u_traj[ii,jj] = K@traj[ii].y[:,jj]
#     plt.plot(t_vec,u_traj[ii,:])
# plt.hlines(y=[umax, -umax], xmin=0, xmax=100, colors=['r', 'r'], linestyles='--', label=f'u_max = {umax}')
# plt.ylabel('u (input)')
# plt.xlabel('Time (s)')
# plt.grid(True)
# plt.legend()
# plt.show()