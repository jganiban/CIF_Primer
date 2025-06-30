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

# def f(x, u):
#     return A @ x + B @ u 


# Define time vector for discretization.
N = 20
dti = 0.5
T = N*dti
print(T)

# SCP solveing parameters
max_iter = 15
eps = 1e-4

# Define initial conditions and terminal state
x0 = np.array([[0],[0],[0],[0]])
xT = np.array([[10],[0],[0],[0]])

obs_c = np.array([4,0])
obs_r = 0.5
safe_dist = 0.5
trust_radius = 1.0
# obs = np.concatenate([obs_c],[obs_r])

# Initialize trajectory with straight line guess
x_traj = np.linspace(x0[0],xT[0],N)
y_traj = np.linspace(x0[2],xT[2],N)
vx_traj = np.zeros(N)
vy_traj = np.zeros(N)
u_traj = np.zeros((N-1,2))

Ad = np.eye(4) + dti * A
Bd = dti * B


def solve_scp(x_guess, max_iter=30):
    # Loop over SCP iterations
    for ii in range(max_iter):
        # Define time step for iteration
        dt = T/N

        # Define state, control, and time cvxpy variables
        x = cp.Variable((N,4))
        u = cp.Variable((N-1,2))
        # T_var = cp.Variable(nonneg = True)

        # Define constraints
        constraints = []
        constraints += [x[0,:] == x0.flatten()] # Initial state constraint
        constraints += [x[N-1,:] == xT.flatten()] # Terminal state constraint
        # constraints += [x[N-1, [0,2]] == xT[[0,2]].flatten()]  # Only match position

        # nu = cp.Variable(N-1,nonneg=True)


        for kk in range(N-1):
            # Integrate dyanmics over one time step
            constraints += [x[kk+1,:] == Ad @ x[kk,:] + Bd @ u[kk,:]]


            # Linearized obstacle constraint
            pos = x_guess[kk,[0,2]]
            vec = pos - obs_c
            norm_vec = np.linalg.norm(vec)
            a = vec / norm_vec
            r_safe = obs_r + safe_dist
            lhs = a @ cp.hstack([x[kk,0], x[kk,2]])
            rhs = a @ pos + r_safe - norm_vec
            constraints += [lhs >= rhs]

            # Control bound
            constraints += [cp.norm(u[kk,:], 2) <= 5.0]

        objective = cp.Minimize(cp.sum_squares(u))
        prob = cp.Problem(objective,constraints)

        prob.solve(solver=cp.MOSEK)
        print(f"Solver status: {prob.status}")
        print(f"Solver status: {prob.status}")
        if x.value is None:
            print("Infeasible problem at iteration", ii+1)
            break


        x_guess = x.value

        print(f"Iteration {ii+1}")
    return x_guess, u.value

x_vals = np.linspace(x0[0], xT[0], N).flatten()
y_vals = 1 * np.sin(np.pi * x_vals / x_vals[-1])  # offset trajectory
x_guess = np.vstack([
    x_vals,
    np.zeros(N),  # vx
    y_vals,
    np.zeros(N)   # vy
]).T


x_opt, u_opt = solve_scp(x_guess, max_iter)
x_traj = x_opt[:,0]
vx_traj = x_opt[:,1]
y_traj = x_opt[:,2]
vy_traj = x_opt[:,3]
u_traj = u_opt

plt.figure()
for obs in obs_c:
    circle = plt.Circle((obs_c[0], obs_c[1]), obs_r, color='r', alpha=0.5)
    plt.gca().add_patch(circle)

plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.title("Obstacle Avoiding Trajectory")

plt.plot(x_traj,y_traj)
plt.plot(x_guess[:,0],x_guess[:,2])
plt.show()









