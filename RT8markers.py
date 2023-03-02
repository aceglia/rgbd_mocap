import biorbd
import numpy as np
from pyomeca import Markers
from pyomeca import Rototrans
import scipy.io as sio
import os.path
import matplotlib.pyplot as plt
from casadi import MX, SX, Function, vertcat, nlpsol
import csv

def RT(transX, transY, angleX):
    Rototrans = np.array((
        (np.cos(angleX), -np.sin(angleX), transX),
        (np.sin(angleX), np.cos(angleX), transY),
        (0, 0, 1)))
    return Rototrans


J = 0
w = []
lbw = []
ubw = []

transX = SX.sym('transX', 1)
transY = SX.sym('transY', 1)

angleX = SX.sym('angleX', 1)


for i in range(3):
    lbw.append(-10)
    ubw.append(10)
for i in range(3):
    lbw.append(-np.pi)
    ubw.append(np.pi)

Rototrans_sym = RT(transX, transY, angleX)



c = 0
for i in range(3):
    for l in range(n_mark):
        for k in range(n_frames):
            if k == 0:
                if np.isnan(marker_treat[i, l, k]) == True:
                    while np.isnan(marker_exp[i, l, k + c]) == True:
                        c += 1
                    marker_treat[i, l, k] = marker_exp[i, l, k + c]
                    c = 1
            else:
                c = 0
                if np.isnan(marker_treat[i, l, k]) == True:
                    while np.isnan(marker_exp[i, l, k + c]) == True:
                        c += 1
                    yb = marker_exp[i, l, k + c]
                    xb = k + c
                    ya = marker_treat[i, l, k - 1]
                    xa = k - 1
                    c = (yb - ya) / (xb - xa)
                    d = ya - c * xa
                    marker_treat[i, l, k] = c * k + d
                    c = 1

# for i in range(3):
#     marker_treat[:,8,210+i] = marker_treat[:,8,209]
marker_treat[3,:,:] = [1]

weight = 1000
# rt = RT((Rototrans(transX, transY,transZ, angleX, angleY, angleZ)))
# marker_rotate = Markers.from_rototrans(marker_exp, rt)
for i in range(marker_model.shape[1]):
    if i in (0,1,4,8,9):
        J = J + sum((marker_model[:,i,0] - np.sum(Rototrans_sym * marker_treat[:,i,0], axis=1))**2) * weight

    else:
        J = J + sum((marker_model[:, i, 0] - np.sum(Rototrans_sym * marker_treat[:, i, 0], axis=1)) ** 2)

w = vertcat(transX,transY,angleX)
prob = {'f':J, 'x':w}
options = {'ipopt.hessian_approximation':"exact",
           # 'ipopt.tol':1e-10,'ipopt.dual_inf_tol':1e-15
           }
solver = nlpsol('solver', 'ipopt', prob, options)
w0 = np.zeros((6,1))
# w0[0]= marker_model[0,0,0] - marker_treat[0,0,0]
# w0[1]= marker_model[1,0,0] - marker_treat[1,0,0]
# w0[2]= marker_model[2,0,0] - marker_treat[2,0,0]

solve = solver(x0 = w0, lbx=-1000, ubx=10000)
w_opt = solve['x']
print(w_opt)
rt = Rototrans((RT(w_opt[0],w_opt[1],w_opt[2],w_opt[3],w_opt[4],w_opt[5])))
# rt = Rototrans((RT(0.841114, 0.103637, -0.596597, -0.458244, 0.145129, 3.21253)))
marker_rotate = Markers.from_rototrans(marker_treat, rt)
# Data for three-dimensional scattered points
ax = plt.axes(projection='3d')
zdata = marker_rotate[2,:,0]
xdata = marker_rotate[0,:,0]
ydata = marker_rotate[1,:,0]
ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
zdata = marker_model[2,:,0]
xdata = marker_model[0,:,0]
ydata = marker_model[1,:,0]
ax.scatter3D(xdata, ydata, zdata, edgecolors="red")
ax.view_init(60, 35)
plt.show()

# nb_frame = range(1,marker_rotate.shape[2]+1)
# anb_frame = np.ndarray((1,marker_rotate.shape[2]))
# for i in range(len(nb_frame)):
#      anb_frame[:,i] = nb_frame[i]
# t = np.linspace(0, 0.24, num=marker_rotate.shape[2]).reshape((1,marker_rotate.shape[2]))
# marker_treat = np.reshape(marker_rotate[:-1,:,:], (marker_rotate.shape[1]*3, marker_rotate.shape[2]), order="F")
# marker_treat = np.concatenate((anb_frame, t, marker_treat), axis=0)
#
# with open(os.path.split(os.path.dirname(__file__))[0]+'/model_scaling/marker_rot.trc', "w") as markers:
#      writer = csv.writer(markers, delimiter='\t')
#      writer.writerows(marker_treat.T)


# #
# rt = Rototrans((RT(w_opt[0],w_opt[1],w_opt[2],w_opt[3],w_opt[4],w_opt[5])))
# # rt = Rototrans((RT(0.841114, 0.103637, -0.596597, -0.458244, 0.145129, 3.21253)))
# marker_rotate = Markers.from_rototrans(marker_treat, rt)
# dic = {"marker_rot" : marker_rotate, "marker_without_rot": marker_treat}
# sio.savemat("./Sujet_5/marker_hirozon_co.mat", dic)