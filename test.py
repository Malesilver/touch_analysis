from scipy import interpolate
import numpy as np
import pandas as pd

# x = np.array(range(2))
# y = np.array(range(2))
# a = np.array([[0, 1], [2, 3]])
#
# f = interpolate.interp2d(x, y, a, kind='linear')
#
# xnew = np.linspace(0, 2, 4)
# ynew = np.linspace(0, 2, 4)
# znew = f(xnew, ynew)
# print(znew)
#
#
# # a = [1,2,3,4]
# # print(a[0:4])
# # a =20*np.log10(1)
# # print(a)
#
df = pd.DataFrame(index=['grid_data', 'line_data'],
                  data={
                      'raw data (min)': [1, 2],
                      "dwadaw": [4, 6]})

import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns

# create the figure
# fig = plt.figure(figsize=[12.99, 8.49])
# ax = fig.add_subplot(111)
# im = sns.heatmap(data=np.random.random((10,10)), cbar = False )
# # im = ax.imshow(np.random.random((50,50)))
# plt.show(block=False)
#
# # draw some data in loop
# for i in range(10000):
#     # wait for a second
#     # time.sleep(0.0001)
#     # replace the image contents
#     im = sns.heatmap(data=np.random.random((10,10)), cbar = False)
#     # redraw the figure
#
#     fig.canvas.draw()
#     fig.canvas.flush_events()

import seaborn as sns
import copy

a = np.random.randint(1, 10, (4, 4))
#
# mask = a>3
# ma = np.ma.masked_array(a,~mask)
# print(ma.fill(10))
# a= a.astype(float)
# a[2][1] = np.NaN
# # print(np.where(np.isnan(a)))
#
# plt.figure()
# sns.heatmap(a)
# plt.show()

print(20 * np.log10(850 / 15))
# print(638/192)
# target_ratio = 10**(30/20)
#
# signal = 350
# print(target_ratio)
#
# print(signal / target_ratio)

signals = [300, 500, 1000]

# for i in range(20,50,5):
#     for signal in signals:
#         target_ratio = 10 ** (i / 20)
#         print(f"expected SNR (dB) = {i}, expected signal/noise ratio = {round(target_ratio,1)}, signal = {signal}, noise = {int(signal / target_ratio)} ")

bb = {"123": {
    "456": "hi"
}}

print(bb.get("456","hhhhh").get("456",46666))
