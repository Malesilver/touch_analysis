import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
import  time
#
# # Constants
# J = 1
# h = 1
# kbT = 1
# beta = 1
#
# # Grid
# L = 20  # Dimensions
# N = L ** 2  # Total number of grid points
#
# # Initial configuration
# spins = 2 * np.random.randint(2, size=(L, L)) - 1
#
# E = []
# i = 0
#
# plt.ion()
# plt.figure()
# plt.show()
#
# while i < 100000:
#     for i in range(1, N):
#         i += 1
#         s = tuple(npr.randint(0, L, 2))  # Random initial coordinate
#
#         # x and y coordinate
#         (sx, sy) = s
#         # Periodic boundary condition
#         sl = (sx - 1, sy)
#         sr = ((sx + 1) % L, sy)
#         sb = (sx, sy - 1)
#         st = (sx, (sy + 1) % L)
#         # Energy
#         E = spins[s] * (spins[sl] + spins[sr] + spins[sb] + spins[st])
#         if E <= 0:  # If negative, flip
#             spins[s] *= -1
#         else:
#             x = np.exp(-E / kbT)  # If positve, check condition
#             q = npr.rand()
#             if x > q:
#                 spins[s] *= -1
#     # Plot (heatmap)
#     plt.clf()
#     sns.heatmap(spins, cmap='magma')
#     plt.pause(0.1)


# create the figure
fig = plt.figure(figsize=[12.99, 8.49])
ax = fig.add_subplot(111)
# im = sns.heatmap(data=np.random.random((50,50)), annot=True, fmt='.0f', vmin=0, vmax=1000)
# ax.imshow(np.random.random((50,50)))
plt.show(block=False)

# draw some data in loop
for i in range(10000):
    # wait for a second
    # time.sleep(0.01)
    # replace the image contents
    im = sns.heatmap(data=np.random.random((10, 10)),cbar = False)
    # redraw the figure

    plt.pause(0.000001)
    ax = plt.cla()