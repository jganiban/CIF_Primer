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
from sympy import symbols, lambdify
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import simplify, cos, sin, atan2, asin, exp, tan
from scipy.linalg import sqrtm

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

def deg2rad(angle):
    return angle*np.pi/180

# Defining system parameters.
m = 2
# J_B = 0.01 * np.eye(3) # 1st MOI
# Jx, Jy, Jz = np.diag(J_B)

J_B = np.array([[0.29292, 0, 0],[0, 0.29292, 0],[0, 0, 0.0025]])

r_t = 0.25 # gimbal point unit vector
g = 1.625 # unit gravity

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
Fxy_max = 1.5
Fxy_min = -1.5
Fz_max = 5.0
Fz_min = 0.0
tau_max = 0.01
tau_min = -0.01

# # Define constraint directions
# u_con = np.vstack([
#     np.eye(Bm),       # u_i ≤ umax
#     -np.eye(Bm)       # -u_i ≤ -umin,  u_i ≥ umin
# ])  # shape (2*Bm, Bm)

# u_bounds = np.vstack([
#     np.ones((Bm, 1))@np.array([Fxy_max, Fxy_max, Fz_max, tau_max, tau_max, tau_max]).T,    # upper bounds
#     np.ones((Bm, 1))@np.array([Fxy_min, Fxy_min, Fz_min, tau_min, tau_min, tau_min]).T    # lower bounds 
# ]) 

u_bounds = np.array([
    [Fxy_max],     # F_x max
    [Fxy_max],     # F_y max
    [Fz_max],     # F_z max
    [tau_max],    # tau_x max
    [tau_max],    # tau_y max
    [tau_max],    # tau_z max
    [Fxy_max],     # F_x min (negated below)
    [Fxy_max],     # F_y min
    [Fz_min],     # F_z min (note: constraint will be one-sided!)
    [tau_max],    # tau_x min
    [tau_max],    # tau_y min
    [tau_max]     # tau_z min
])

u_con = []
u_bounds_trimmed = []

# Fx, Fy: symmetric bounds
for i in range(2):
    ei = np.zeros((Bm,))
    ei[i] = 1
    u_con.append(ei)
    u_bounds_trimmed.append(1.5)
    u_con.append(-ei)
    u_bounds_trimmed.append(1.5)

# Fz: one-sided (upper bound only)
ei = np.zeros((Bm,))
ei[2] = 1
u_con.append(ei)
u_bounds_trimmed.append(5.0)

# Tau: symmetric bounds
for i in range(3, 6):
    ei = np.zeros((Bm,))
    ei[i] = 1
    u_con.append(ei)
    u_bounds_trimmed.append(0.01)
    u_con.append(-ei)
    u_bounds_trimmed.append(0.01)

u_con = np.array(u_con)
u_bounds_trimmed = np.array(u_bounds_trimmed)

# Define state constraints.
r_max = 8.0
r_min = -8.0
v_max = 2.0
v_min = -2.0
phimax,thetamax,psimax = deg2rad(30),deg2rad(30),deg2rad(30)
phimin,thetamin,psimin = -deg2rad(30),-deg2rad(30),-deg2rad(30)
pmax,qmax,rmax = deg2rad(45),deg2rad(45),deg2rad(45)
pmin,qmin,rmin = -deg2rad(45),-deg2rad(45),-deg2rad(45)

# Construct state constraint matrix and bounds
state_con = []
state_bounds = []

# Position constraints (|r| <= 8)
for i in range(3):  # x, y, z
    ei = np.zeros((12,))
    ei[i] = 1
    state_con.append(ei)
    state_bounds.append(r_max)
    state_con.append(-ei)
    state_bounds.append(r_max)

# Velocity constraints (|v| <= 2)
for i in range(3, 6):  # vx, vy, vz
    ei = np.zeros((12,))
    ei[i] = 1
    state_con.append(ei)
    state_bounds.append(v_max)
    state_con.append(-ei)
    state_bounds.append(v_max)

# Euler angle constraints (30 deg)
angle_limits = np.deg2rad(30)
for i in range(6, 9):  # phi, theta, psi
    ei = np.zeros((12,))
    ei[i] = 1
    state_con.append(ei)
    state_bounds.append(phimax)
    state_con.append(-ei)
    state_bounds.append(phimax)

# Body rate constraints (±45°/s)
rate_limits = np.deg2rad(45)
for i in range(9, 12):  # p, q, r
    ei = np.zeros((12,))
    ei[i] = 1
    state_con.append(ei)
    state_bounds.append(pmax)
    state_con.append(-ei)
    state_bounds.append(pmax)

# Convert to arrays
state_con = np.array(state_con)
state_bounds = np.array(state_bounds)

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
gamma = np.array([ 0.25, 0.25, 0.1])

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
# for jj in range(len(u_bounds)):
#     c_i = u_con[jj, :].reshape((1, Bm))  
#     eq_bias = c_i @ u_eq_vec  
#     d_i_val = u_bounds[jj, 0]**2 - float(eq_bias @ eq_bias.T)

#     Yci = c_i @ Y                         
#     d_i = cp.Constant([[u_bounds[jj,0]**2]])         

#     uLMI = cp.bmat([
#         [d_i,     Yci],    
#         [Yci.T,    Q]         
#     ])
#     constraints += [uLMI >> 0]

for jj in range(len(u_bounds_trimmed)):
    c_i = u_con[jj, :].reshape((1, Bm))        # row vector
    eq_bias = c_i @ u_eq_vec                   # feedforward shift
    d_val = u_bounds_trimmed[jj]**2 - float(eq_bias @ eq_bias.T)  # scalar

    d_i = cp.Constant([[d_val]])               # 1x1
    Yci = c_i @ Y                              # 1x12

    uLMI = cp.bmat([
        [d_i,     Yci],    
        [Yci.T,   Q]
    ])
    constraints += [uLMI >> 0]

# Set up LMI constraint on states.
for ii in range(len(state_bounds)):
    a_i = state_con[i].reshape((1,12))
    b_i = cp.Constant([[state_bounds[ii]**2]])
    LMI = cp.bmat([
        [b_i, a_i @ Q],
        [Q @ a_i.T, Q]
    ])
    constraints += [LMI >> 0]


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

# # Plot rx, ry
# plt.figure()
# plt.plot(ellipse_points[0, :], ellipse_points[1, :])
# plt.gca().set_aspect('equal')
# plt.title("Ellipsoidal Level Set from Q")
# plt.xlabel("rx")
# plt.ylabel("ry")
# plt.grid(True)
# plt.show()


# ------------------------------------------------------------------
# Section 5: Simulate dynamics with IC and optimal K

from scipy.integrate import solve_ivp

def closed_loop_dynamics(t, x, A, B, K, u_eq):
    u_eq_vec = np.array(u_eq).flatten()
    xi =  K@x
    xdot = A@x + B@xi
    return xdot

x_sym = x
u_sym = u
f_func = lambdify((x_sym, u_sym), f, modules='numpy')

def closed_loop_nonlinear_dynamics(t, x, K, u_eq):
    u_eq_vec = np.array(u_eq).flatten()
    u = u_eq_vec + K @ x  # feedback control
    # u = np.clip(u, umin, umax)  # enforce actuator limits
    return np.array(f_func(x, u)).flatten()

# Sample initial conditions evenly from the ellipsoid boundary
initial_conditions = []
num_traj = 100
for ii in range(num_traj):
    z = np.random.randn(12)
    z /= np.linalg.norm(z)
    x0 = sqrtm(Q.value).real@z
    # x0[6:12] = 0.0
    # x0 = np.zeros((12,))

    initial_conditions.append(x0)

t_span = (0, 50)
t_vec = np.linspace(*t_span, 100)

traj = []
u_traj = np.empty((len(initial_conditions),len(t_vec)))
for ii in range(len(initial_conditions)):
    # traj.append(solve_ivp(closed_loop_dynamics, t_span, initial_conditions[ii],'RK45',t_eval=t_vec, args=(A, B, K, u_eq)))
    traj.append(solve_ivp(closed_loop_nonlinear_dynamics, t_span, initial_conditions[ii],'RK45',t_eval=t_vec, args=(K, u_eq)))



# Overlay the trajectory in state space with the ellipsoid
plt.figure(figsize=(6, 6))
plt.plot(ellipse_points[0,:], ellipse_points[1,:], label="Ellipsoid")
for ii in range(len(initial_conditions)):
    plt.plot(traj[ii].y[0], traj[ii].y[1])
    if ii == 1:
        plt.scatter(traj[ii].y[0,0], traj[ii].y[1,0], label="Initial State")
    else:
        plt.scatter(traj[ii].y[0,0], traj[ii].y[1,0])
plt.xlabel("rx")
plt.ylabel("ry")
plt.title("Trajectory within Control Invariant Ellipsoid")
plt.grid(True)
plt.legend()
plt.xlim([-np.max(ellipse_points[0,:])-10, np.max(ellipse_points[0,:])+10])
plt.ylim([-np.max(ellipse_points[1,:])-10, np.max(ellipse_points[1,:])+10])

plt.show()

# Scatter terminal state with the ellipsoid
# plt.figure(figsize=(6, 6))
# plt.plot(ellipse_points[0,:], ellipse_points[1,:], label="Ellipsoid")
# for ii in range(len(initial_conditions)):
#     plt.scatter(traj[ii].y[0,-1], traj[ii].y[1,-1])
# plt.xlabel("rx")
# plt.ylabel("ry")
# plt.title("Trajectory within Control Invariant Ellipsoid")
# plt.grid(True)
# plt.legend()
# # plt.xlim([-np.max(ellipse_points[0,:])-10, np.max(ellipse_points[0,:])+10])
# # plt.ylim([-np.max(ellipse_points[1,:])-10, np.max(ellipse_points[1,:])+10])

# plt.show()

# Extract Q and get the top-left 3x3 submatrix for position
Q_val = Q.value
Q_pos = Q_val[:3, :3]  # Assuming x[0:3] = [rx, ry, rz]

# Eigen-decomposition of Q_pos
eigvals, eigvecs = np.linalg.eigh(Q_pos)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Generate unit sphere
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
x_s = np.outer(np.cos(u), np.sin(v))
y_s = np.outer(np.sin(u), np.sin(v))
z_s = np.outer(np.ones_like(u), np.cos(v))

# Stack and transform each point on the unit sphere
sphere = np.stack((x_s, y_s, z_s), axis=0).reshape(3, -1)  # shape (3, N)
scale = np.sqrt(eigvals)[:, None]  # shape (3,1)
ellipsoid = eigvecs @ (scale * sphere)  # shape (3, N)

# Reshape back to (N, N) for plotting
x_e = ellipsoid[0, :].reshape(x_s.shape)
y_e = ellipsoid[1, :].reshape(y_s.shape)
z_e = ellipsoid[2, :].reshape(z_s.shape)

# # Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('rx')
# ax.set_ylabel('ry')
# ax.set_zlabel('rz')
# ax.set_title('3D Control Invariant Ellipsoid in Position Space')
# ax.set_box_aspect([1,1,1])  # Equal aspect ratio
# plt.show()

# Plot 3-D position
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_e, y_e, z_e, rstride=1, cstride=1, alpha=0.2)
for ii in range(len(initial_conditions)):
    plt.plot(traj[ii].y[0], traj[ii].y[1], traj[ii].y[2])
    # plt.scatter(traj[ii].y[0,0], traj[ii].y[1,0], traj[ii].y[2,0], label="Initial State")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_aspect('auto')
plt.show()

state_labels = [
    'rx', 'ry', 'rz',
    'vx', 'vy', 'vz',
    'phi', 'theta', 'psi',
    'p', 'q', 'r'
]

def plot_states_vs_time(traj, state_labels):
    # t = traj.t
    # x = traj.y  # shape (12, len(t))

    fig, axs = plt.subplots(4, 3, figsize=(15, 10))
    axs = axs.flatten()

    for j in range(len(traj)):
        for i in range(12):
            axs[i].plot(traj[j].t, traj[j].y[i, :])
            axs[i].set_title(state_labels[i])
            axs[i].set_xlabel('Time [s]')
            axs[i].set_ylabel(state_labels[i])
            axs[i].grid(True)

    plt.tight_layout()
    plt.suptitle('Rocket States Over Time', y=1.02)
    plt.show()

plot_states_vs_time(traj, state_labels)

# Plot input u vs t
# for ii in range(num_traj):
#     for jj in range(len(t_vec)):
#         u_traj[ii,jj] = K@traj[ii].y[:,jj]
#     plt.plot(t_vec,u_traj[ii,:])
# plt.hlines(y=[umax, -umax], xmin=0, xmax=100, colors=['r', 'r'], linestyles='--', label=f'u_max = {umax}')
# plt.ylabel('u (input)')
# plt.xlabel('Time (s)')
# plt.grid(True)
# plt.legend()
# plt.show()
# 
# from numpy.linalg import norm

# C1 = np.hstack((np.zeros((6,6)),np.vstack((np.eye(3),np.zeros((3,3)))), np.zeros((6,3))))
# D1 = np.hstack((np.vstack((np.zeros((3,3)),np.eye(3))),np.zeros((6,3))))
# C2 = np.hstack((np.zeros((4,6)),np.array([[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]])))
# D2 = np.zeros((4,6))
# C3 = np.hstack((np.zeros((3,9)),np.eye(3)))
# D3 = np.zeros((3,6))

# C = np.vstack((C1,C2,C3))
# D = np.vstack((D1,D2,D3))

# def compute_q(x, u):
#     return C @ x + D @ u

# def compute_p(x, u):
#     q = compute_q(x, u)
#     dx_nl = np.array(f_func(x, u)).flatten()
#     dx_lin = A @ x + B @ u
#     resid = dx_nl - dx_lin
#     return E.T @ resid  # reconstruct the implied p(q)

# filtered_ratios = []

# for _ in range(1000):
#     x = sqrtm(Q.value).real @ np.random.randn(12)
#     u = u_eq_vec + K @ x
#     q = np.array(compute_q(x, u), dtype=np.float64).flatten()
#     p = np.array(compute_p(x, u), dtype=np.float64).flatten()
#     norm_q_blocks = [norm(q[0:6]), norm(q[6:10]), norm(q[10:13])]
#     norm_p_blocks = [norm(p[0:3]), norm(p[3:6]), norm(p[6:9])]
#     try:
#         ratio = [norm_p_blocks[i] / (norm_q_blocks[i] + 1e-6) for i in range(3)]
#         if all(nq > 1e-3 for nq in norm_q_blocks):  # filter
#             filtered_ratios.append(ratio)
#     except ZeroDivisionError:
#         continue

# filtered_ratios = np.array(filtered_ratios)
# gamma_max = np.percentile(filtered_ratios, 95, axis=0)
# gamma_safe = 0.9 * gamma_max
# print("Tuned gamma:", gamma_safe)