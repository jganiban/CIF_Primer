# ------------------------------------------------------------------
# Lesson3.py
# This module implements an example of computing a trajectory that
# avoids obstacles using SCP. Uses the same set of dynamic 
# equations as Lesson 1, but expanded to 2D.
#
# Justin Ganiban
# 5/7/25
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Section 1: Defining the problem.
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Define LTI system.
A = np.array([
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
    ])
An, Am = A.shape

B = np.array([
    [0, 0],
    [1, 0],
    [0, 0],
    [0, 1]
    ])
Bn, Bm = B.shape

def f(x, u): return A@x+B@u

# Define time vector for discretization.
N = 50
dti = 0.1
T = N*dti

# SCP solveing parameters
max_iter = 15
eps = 1e-4

# Define initial conditions and terminal state
x0 = np.array([[0],[0],[0],[0]])
xT = np.array([[10],[0],[0],[0]])

# Define the obstacle position and radius
obs_pos = np.array([[3.0,2.0], [7.0,-2.0]])
obs_rad = 0.5
# obs = np.array([obs_pos[0,:],obs_rad],[obs_pos[0,:],obs_rad])

# Initialize trajectory with straight line guess
x_traj = np.linspace(x0[0],xT[0],N)
y_traj = np.linspace(x0[2],xT[2],N)
vx_traj = np.zeros(N)
vy_traj = np.zeros(N)
u_traj = np.zeros((N-1,2))

# Loop over SCP iterations
for ii in range(max_iter):
    # Define time step for iteration
    dt = T/N

    # Define state, control, and time cvxpy variables
    x = cp.Variable((N,4))
    u = cp.Variable((N-1,2))
    T_var = cp.Variable(nonneg = True)

    # Define constraints
    constraints = []
    constraints += [x[0,:] == x0] # Initial state constraint
    constraints += [x[N-1,:] == xT] # Terminal state constraint


    for kk in range(N-1):
        # Integrate dyanmics over one time step
        xk = x[kk,:]
        uk = u[kk,:]

        k1 = f(x[kk], u[kk])
        k2 = f(x[kk] + 0.5 * dt * k1, u[kk])
        k3 = f(x[kk] + 0.5 * dt * k2, u[kk])
        k4 = f(x[kk] + dt * k3, u[kk])
        xkp1 = x[kk] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        constraints += [x[kk+1,:] == xkp1]

        # Obstacle constraints
        px, py = x_traj[kk], y_traj[kk]
        for jj in range(len(obs_pos)):
            diff = np.array([px, py]) - obs_pos[jj,:]
            dist = np.maximum(np.linalg.norm(diff), 1e-2)
            grad = diff / dist
            constraints += [grad[0] * (x[kk, 0] - px) + grad[1] * (x[kk, 2] - py) >= obs_rad - dist + 0.1]
        
    objective = cp.Minimize(T_var)
    prob = cp.Problem(objective,constraints)

    prob.solve(solver=cp.CLARABEL)

    x_val = x.value
    x_traj = x_val[:,0]
    vx_traj = x_val[:,1]
    y_traj = x_val[:,2]
    vy_traj = x_val[:,3]
    u_traj = u.value
    T_new = T_var.value

    print(f"Iteration {iter+1}: Time = {T_new:.3f}")
    if iter > 0 and abs(T_new - T) < eps:
        print("Converged.")
        break

    T = T_new

plt.figure()
plt.plot(x_traj,y_traj)









