# ------------------------------------------------------------------
# Lesson4.py
# This module implements an example of computing a control invariant
# ellipsoidal set with uncertainty dynamics representing input saturation
# and drag for a discrete-time linear system using cvxpy.
#
# Justin Ganiban
# 5/15/25
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
E = np.array([[0, 0],[1, 1]])
En, Em = E.shape

# Define velocity and input constraints - |v| <= vmax, |u| <= umax.
vmax = 1.0
v_con = np.array([[0, 1],[0, -1]])
v_bounds = np.array([[vmax, -vmax]])

umax = 10.0
u_con = np.array([[1],[-1]])
u_bounds = np.array([[umax], [-umax]])


# Define Uncertainty model: p = sat(a)-a, q = Cx+Du = a, norm(p)<=norm(q)
def saturation(value, lower_bound, upper_bound):
    if value > upper_bound:
        return upper_bound
    elif value < lower_bound:
        return lower_bound
    else:
        return value
    
cd = 1.6
L = np.array([[0.8, 0],[0, cd*vmax]])
C = np.array([[0, 0],[0, 1]])
D = np.array([[1],[0]])

# Define the decision variables.
Q = cp.Variable((An, An),symmetric=True)
Y = cp.Variable((Bm, Bn))
nu1 = cp.Variable()
nu2 = cp.Variable()
nu = cp.diag(cp.vstack([nu1,nu2]))

# Define the decay rate that bounds the Lyapunov function.
alpha = 0.2

# ------------------------------------------------------------------
# Section 2: Set up constraints for cvxpy.
constraints = []

# Define the Lyapunov LMI constraint.
lyap_con = cp.bmat([
    [A@Q + B@Y + Q@A.T + Y.T@B.T + alpha*Q,   nu@E,                             (C@Q+D@Y).T],
    [nu@E.T,                                  cp.reshape(-nu, (2, 2)),          cp.Constant([[0, 0],[0, 0]])],
    [C@Q+D@Y,                                 cp.Constant([[0, 0],[0, 0]]),     cp.reshape(-nu@(L**(-1)**2), (2, 2))]
])
constraints += [lyap_con << 0]
constraints += [nu >> 1e-4]

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
# objective = cp.Maximize(cp.log_det(Q))
objective = cp.Minimize(cp.tr_inv(Q))

# ------------------------------------------------------------------
# Section 3: Solve the problem.
prob = cp.Problem(objective, constraints)

print("----------------------------------")
prob.solve(solver=cp.CLARABEL, verbose=True)
print("solver status : " + prob.status)
print("solve time    :" + str(prob.solver_stats.solve_time))
# print("cost          :" + str(prob.objective.value))
print("----------------------------------")

# Back out the optimal control gain K.
K = Y.value @ np.linalg.inv(Q.value)

# ------------------------------------------------------------------
# Section 4: Extracting the Ellipse and lower level set
# Extract the eigenvalues and eigenvectors of Q.
eigvals, eigvecs = np.linalg.eig(Q.value)

# Create an ellipse from the eigenvalues and eigenvectors.
theta = np.linspace(0, 2 * np.pi, 100)
ellipse_points = np.array([np.cos(theta), np.sin(theta)])

# Scale ellipse points by the square root of the eigenvalues.
ellipse_points_scaled = np.sqrt(eigvals) * ellipse_points.T
ellipse_points_scaled_low = 0.5*ellipse_points_scaled

# Rotate ellipse points by the eigenvectors.  
ellipse_points_rotated = ellipse_points_scaled @ eigvecs.T
ellipse_points_rotated_low = ellipse_points_scaled_low @ eigvecs.T

# # Plot the ellipsoid.
plt.figure(figsize=(6, 6))
plt.plot(ellipse_points_rotated[:, 0], ellipse_points_rotated[:, 1], label="Ellipsoid")

# # Plot vmax constraint.
plt.hlines(y=[vmax, -vmax], xmin=-2000000, xmax = 200000, colors=['r','r'],linestyles=['--','--'], label=f'v_max = {vmax}')

# # Add axes.
plt.axhline(0, color='black',linewidth=1)
plt.axvline(0, color='black',linewidth=1)

# # Adjust aspect ratio and labels.
plt.xlim([-np.max(ellipse_points_rotated[:, 0]), np.max(ellipse_points_rotated[:, 0])])
plt.title("Control Invariant Ellipsoid with vmax constraint")
plt.xlabel("x1 (position)")
plt.ylabel("x2 (velocity)")
plt.grid(True)
plt.legend()
plt.show()

# ------------------------------------------------------------------
# Section 5: Simulate dynamics with IC and optimal K

from scipy.integrate import solve_ivp
import random

def closed_loop_dynamics(t, x, A, B, K, E):
    u = K@x
    p = np.array([[0], [cd*vmax*x[1]]])
    u = saturation(u,-2.0,2.0)
    # xdot = A@x + B@u + E*p
    xdot = A@x + np.squeeze(B*u) - np.squeeze(E@p)

    return xdot

# Sample initial conditions evenly from the ellipsoid boundary
initial_conditions = []
num_traj = 25
for ii in range(num_traj):
    x0p = random.choice(ellipse_points_rotated_low[:,0])
    x0_idx = np.where(ellipse_points_rotated_low[:,0] == x0p)[0]
    x0v = ellipse_points_rotated_low[x0_idx,1]
    x0 = np.array([x0p, x0v[0]])
    initial_conditions.append(x0)

for ii in range(num_traj):
    x0p = random.choice(ellipse_points_rotated[:,0])
    x0_idx = np.where(ellipse_points_rotated[:,0] == x0p)[0]
    x0v = ellipse_points_rotated[x0_idx,1]
    x0 = np.array([x0p, x0v[0]])
    initial_conditions.append(x0)

t_span = (0, 100)
t_vec = np.linspace(*t_span, 500)

traj = []
u_traj = np.empty((len(initial_conditions),len(t_vec)))
for ii in range(len(initial_conditions)):
    traj.append(solve_ivp(closed_loop_dynamics, t_span, initial_conditions[ii],'RK45',t_eval=t_vec, args=(A, B, K, E)))

# Overlay the trajectory in state space with the ellipsoid
plt.figure(figsize=(6, 6))
plt.plot(ellipse_points_rotated[:, 0], ellipse_points_rotated[:, 1], label="Ellipsoid")
plt.plot(ellipse_points_rotated_low[:, 0], ellipse_points_rotated_low[:, 1], label="0.5 Level Set")
for ii in range(len(initial_conditions)):
    plt.plot(traj[ii].y[0], traj[ii].y[1])
plt.hlines(y=[vmax, -vmax], xmin=-2000000, xmax=2000000, colors=['r', 'r'], linestyles='--', label=f'v_max = {vmax}')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel("x1 (position)")
plt.ylabel("x2 (velocity)")
plt.title("Trajectory within Control Invariant Ellipsoid")
plt.grid(True)
plt.legend()
plt.xlim([-np.max(ellipse_points_rotated[:, 0])-4, np.max(ellipse_points_rotated[:, 0])+4])
plt.show()

# Plot input u vs t
u_sat = np.empty((len(initial_conditions),len(t_vec)))
for ii in range(2*num_traj):
    for jj in range(len(t_vec)):
        u_traj[ii,jj] = K@traj[ii].y[:,jj]
        u_sat[ii,jj] = saturation(u_traj[ii,jj],-2,2)
    # plt.plot(t_vec,u_traj[ii,:])
    plt.plot(t_vec,u_sat[ii,:])
plt.hlines(y=[umax, -umax], xmin=0, xmax=100, colors=['r', 'r'], linestyles='--', label=f'u_max = {umax}')
plt.ylabel('u (input)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()
plt.show()