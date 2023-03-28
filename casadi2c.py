'''
This code is heavily inspired by
https://github.com/casadi/casadi/blob/12fa60f676716ae09aa2abe34c1c9b9cf426e68a/docs/examples/python/race_car.py
most credz to that.
'''
from casadi import *

# Degrees to radians conversion
deg2rad = lambda deg: deg*pi/180

N = 100  # Time horizon

# Making optimization object
opti = Opti()

# Declaring optimization variables
x = opti.variable(3, N+1)
x_pos = x[0, :]
y_pos = x[1, :]
theta = x[2, :]
u = opti.variable(2, N)
v = u[0, :]
phi = u[1, :]
T = opti.variable()

# Objective
L = T
opti.minimize(L)

# Model constraints
# dx/dt = f
f = lambda X, U: vertcat(U[0]*cos(X[2]), U[0]*sin(X[2]), U[0]*U[1])

# Time step
dt = T/N

# Fixed step Runge-Kutta 4 integrator
for k in range(N):
    k1 = f(x[:, k],             u[:, k])
    k2 = f(x[:, k] + dt/2 * k1, u[:, k])
    k3 = f(x[:, k] + dt/2 * k2, u[:, k])
    k4 = f(x[:, k] + dt * k3,   u[:, k])
    x_next = x[:, k] + dt/6 * (k1+2*k2+2*k3+k4)
    opti.subject_to(x[:,k+1]==x_next)

# Control signal and time constraint
opti.subject_to(opti.bounded(-1,v,1))
opti.subject_to(opti.bounded(deg2rad(-15), phi, deg2rad(15)))
opti.subject_to(T >= 0)

# Boundary values
# Initial conditions
opti.subject_to(x_pos[0] == 0)
opti.subject_to(y_pos[0] == 0)
opti.subject_to(theta[0] == 0)
opti.set_initial(x_pos,0)
opti.set_initial(y_pos,0)
opti.set_initial(theta,0)
opti.set_initial(T, 1)

# End conditions
opti.subject_to(x_pos[-1] == 0.25)
opti.subject_to(y_pos[-1] == 0.25)
# opti.subject_to(theta[-1] == 0)

# Setup solver and solve
opti.solver('ipopt')
solution = opti.solve()

print(f"x_pos_opt {solution.value(x_pos)}")
print(f"y_pos_opt {solution.value(y_pos)}")
print(solution.value(T))

time_grid = [solution.value(T)/N*k for k in range(N+1)]

# Plotting
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,2, figsize=(8,7))
axs[0,0].plot(solution.value(x_pos), solution.value(y_pos), '--', color='#B6594C', label='Position [m]')
axs[0,0].legend(loc="upper left")
axs[1,0].plot(time_grid, solution.value(theta), '--',color='#90AEB2', label=r'$\theta$ [rad]')
axs[1,0].legend(loc="upper left")
axs[0,1].plot(time_grid, vertcat(DM.nan(1), solution.value(v)), '-', color='#37514D', label='v [m/s]')
axs[0,1].legend(loc="upper left")
axs[1,1].plot(time_grid, vertcat(DM.nan(1), solution.value(phi)), '-', color='#DD8E75', label=r'$\phi$ [rad]')
axs[1,1].legend(loc="upper left")
plt.savefig('2c.png', dpi = 300)
plt.show()