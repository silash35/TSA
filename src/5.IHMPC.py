import os
from control import ss
from ihmpclab.components import IHMPC, ClosedLoop, Model
import numpy as np
from lib.read_data import u_desvio as u_true
from lib.read_data import y_desvio as y_true
import hickle as hkl

ny = y_true.shape[1]
nu = u_true.shape[1]

theta = hkl.load("../export/ARX.hkl")

A = theta[:ny, :].T
B = theta[ny:, :].T
C = np.eye(A.shape[0])
D = np.zeros((C.shape[0], B.shape[1]))

Ts = 1  # sampling time in min
g = ss(A, B, C, D, Ts)

# setting the model objects
plant = Model(g, Ts, "Positional")
controllermodel = Model(g, Ts)

controllermodel.labels["inputs"] = ["F /(ton/h)", "Q /($m^3$/d)"]
controllermodel.labels["outputs"] = ["$T /(C)$", "Flood %"]
controllermodel.labels["time"] = "Time /(min)"

# creating the controller
m = 4  # control horizon
qy = [1, 1]  # output weights
r = [1, 1 / 60]  # input movement weights
sy = 100  # output slack weights
controller = IHMPC(controllermodel, m, qy, r, sy=sy, zone=False)

# Kalman Filter
W = 0.5  # model
V = 0.5  # plant
controller.Kalman(W, V)

# Initial conditions
u0 = [0, 0.1]
x0_controller = [0] * controllermodel.nx
x0_plant = [0] * plant.nx  # dimensions are resolved by validation

# Closedloop
closedloop = ClosedLoop(plant, controller)
closedloop.initialConditions(x0_plant, x0_controller)

# Constraints
umin = [-0.5, -30]  # lower bounds of inputs
umax = [0.5, 30]  # upper bounds of inputs
dumax = [0.2, 10]  # maximum variation of input moves

ysp = [[0, 0], [0.1, -1], [0.1, 1], [0, 0]]  # Set-point values
ysp_change = [10, 50, 100]

tf = 150
closedloop.configPlot.folder = os.path.dirname(os.path.abspath(__file__))
closedloop.configPlot.subfolder = "Stable - Distillation Column - Set-point Tracking"
closedloop.simulation(
    tf, u0=u0, umin=umin, umax=umax, dumax=dumax, spec_change=ysp_change, ysp=ysp
)
