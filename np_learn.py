import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
a=np.random.randint(1,10,(4,4))

# mask = a>3
# ma = np.ma.masked_array(a,~mask)
# print(ma.fill(10))
# a= a.astype(float)
# print(np.nanmean(a))
# a[2][:] = np.NaN
# # print(np.where(np.isnan(a)))
# print(np.nanmean(a))
# plt.figure()
# sns.heatmap(a)
# plt.show()

print(a.mean(axis=0).shape)
plt.figure()
plt.bar(range(len(a)),a.mean(axis=0))
plt.show()
