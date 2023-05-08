import numpy as np
import matplotlib.pyplot as plt

# This works
# my_time_series = dict()
# for L in range(20,50,10):
#    scaling = np.random.randint(100)
#    my_time_series[L] = scaling * np.random.rand(L) + scaling * np.random.rand(L)

# y_normed = {k: (data-np.mean(data))/np.std(data) 
#             for k, data in my_time_series.items()}

# maxlength = max(my_time_series)
# x_interped = {k: np.interp(np.linspace(0, 1, maxlength), 
#                            np.linspace(0, 1, k), data) 
#               for k, data in my_time_series.items()}

# [plt.plot(data) for data in x_interped.values()]
# plt.show()

traj1 = np.array([1, 1, 2, 3, 4])
traj2 = np.array([1, 2, 3])
traj3 = np.array([0, 5, 6, 7, 6, 5, 0])
traj4 = np.array([-1, -2, -2, -2, -3, -3, -2, -2, -1, -1, -1])

trajectories = [traj1, traj2, traj3, traj4]
num_traj = len(trajectories)

keys = ['traj1', 'traj2', 'traj3', 'traj4']

joints = {}
# keys = range(num_traj)
for i in range(len(keys)):
    joints[keys[i]] = trajectories[i]

print(joints)

# joints = {'joint1': traj1, 'joint2': traj2, 'joint3': traj3, 'joint4': traj4}


maxlength = len(joints[max(joints)])

x_interped = {k: np.interp(np.linspace(0, 1, maxlength), np.linspace(0, 1, len(joints[k])), data) for k, data in joints.items()}

[plt.plot(data) for data in x_interped.values()]
plt.show()