"""
This implements the trajopt method for collision avoidance given initial trajectory guess
input:
    trajectory:
        (assuming time is 0.02s)
        state:
        control:
"""

import pytorch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import pickle
STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
MIN_V_1, MAX_V_1 = -6., 6.
MIN_V_2, MAX_V_2 = -6., 6.
MIN_TORQUE, MAX_TORQUE = -4., 4.

MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi

LENGTH = 20.
m = 1.0
lc = 0.5
lc2 = 0.25
l2 = 1.
I1 = 0.2
I2 = 1.0
l = 1.0
g = 9.81
num_dis_pts = 10 # num of points sampled on the acrobot for each link
collision_K = 1.  # parameter for collision
dt = 0.02

def dynamics(state, control):
    '''
    Port of the cpp implementation for computing state space derivatives
    '''
    theta2 = state[:,STATE_THETA_2]
    theta1 = state[:,STATE_THETA_1] - np.pi/2
    theta1dot = state[:,STATE_V_1]
    theta2dot = state[:,STATE_V_2]
    _tau = control[:,0]
    d11 = m * lc2 + m * (l2 + lc2 + 2 * l * lc * torch.cos(theta2)) + I1 + I2
    d22 = m * lc2 + I2
    d12 = m * (lc2 + l * lc * torch.cos(theta2)) + I2
    d21 = d12

    c1 = -m * l * lc * theta2dot * theta2dot * torch.sin(theta2) - (2 * m * l * lc * theta1dot * theta2dot * torch.sin(theta2))
    c2 = m * l * lc * theta1dot * theta1dot * torch.sin(theta2)
    g1 = (m * lc + m * l) * g * torch.cos(theta1) + (m * lc * g * torch.cos(theta1 + theta2))
    g2 = m * lc * g * torch.cos(theta1 + theta2)

    u2 = _tau - 1 * .1 * theta2dot
    u1 = -1 * .1 * theta1dot
    theta1dot_dot = (d22 * (u1 - c1 - g1) - d12 * (u2 - c2 - g2)) / (d11 * d22 - d12 * d21)
    theta2dot_dot = (d11 * (u2 - c2 - g2) - d21 * (u1 - c1 - g1)) / (d11 * d22 - d12 * d21)
    deriv = torch.stack([theta1dot, theta2dot, theta1dot_dot, theta2dot_dot])
    return deriv


def collision_loss(obs, state):
    # given the obs and state, compute the collision loss
    x1 = torch.cos(state[0] - np.pi / 2)
    y1 = torch.sin(state[0] - np.pi / 2)
    dx = torch.cos(state[0] + state[1] - np.pi / 2)
    dx = torch.sin(state[0] + state[1] - np.pi / 2)
    loss = 0.
    for i in range(1,num_dis_pts+1):
        loss_i = (x1*float(i/num_dis_pts) - obs[0])**2 + (y1*float(i/num_dis_pts) - obs[1])**2
        loss_i = 1 / loss_i
        loss += loss_i * collision_K

        loss_i = (x1 + dx*float(i/num_dis_pts) - obs[0])**2 + (y1 + dx*float(i/num_dis_pts) - obs[1])**2
        loss_i = 1 / loss_i
        loss += loss_i * collision_K
    return loss
def dynamics_loss(start, start_control, state, control):
    state_dot = dynamics(state[:-1], control[:-1])
    loss = torch.sum((state_dot*dt + state[:-1] - state[1:])**2)
    dyamics_K = 1.
    # for start
    loss += (start+dt*dynamics(start, start_control)-state[1])**2
    loss = loss * dynamics_K
    return loss
def trajopt(init_state, init_control, obs):
    max_iter = 1000
    state = torch.from_numpy(init_state[1:])
    control = torch.from_numpy(init_control[1:])
    start = torch.from_numpy(init_state[0])  # start is not optimized
    start_control = torch.from_numpy(init_control[0])
    goal = torch.from_numpy(init_state[-1])
    obs = torch.from_numpy(obs)

    state = Variable(state)
    control = Variable(control)
    optimizer = optim.SGD([state, control], lr=0.01)
    for i in range(max_iter):
        optimizer.zero_grad()
        c_loss = 0.
        d_loss = 0.
        f_loss = 0.
        for t in range(len(state)):
            for obs_i in range(len(obs)):
                c_loss += collision_loss(obs[obs_i], state[t])
        d_loss = dynamics_loss(start, start_control, state, control)
        f_loss (state[-1,:2] - goal[:2])**2  # loss for endpoint
        loss = c_loss + d_loss + f_loss
        loss.backward()
        optimizer.step()
        print('iteration %d' % (i))
        print('collision loss: %f' % (c_loss))
        print('dynamics loss: %f' % (d_loss))
        print('endpoint loss: %f' % (f_loss))
        print('total loss: %f' % (loss))



obs_list_total = []
obc_list_total = []
for i in range(2):
    file = open('data/acrobot_simple/obs_%d.pkl' % (i), 'rb')
    obs_list_total.append(pickle.load(file))
    file = open('data/acrobot_simple/obc_%d.pkl' % (i), 'rb')
    obc_list_total.append(pickle.load(file))

obs_idx = 0
p_idx =0
# Create custom system
#obs_list = [[-10., -3.],
#            [0., 3.],
#            [10, -3.]]
obs_list_total = np.array(obs_list_total)
obc_list_total = np.array(obc_list_total)
obs_list = obs_list_total[obs_idx]
obc_list = obc_list_total[obs_idx]
print('generated.')
print(obs_list.shape)


path = open('data/acrobot_simple/%d/path_%d.pkl' % (obs_idx, p_idx), 'rb')
path = pickle.load(path)
controls = open('data/acrobot_simple/%d/control_%d.pkl' % (obs_idx, p_idx), 'rb')
controls = pickle.load(controls)
costs = open('data/acrobot_simple/%d/cost_%d.pkl' % (obs_idx, p_idx), 'rb')
costs = pickle.load(costs)

trajopt(path, controls, costs)
