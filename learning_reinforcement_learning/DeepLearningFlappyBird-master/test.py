import numpy as np
import matplotlib.pyplot as plt

a = np.load('tmp.npy')
print(a.shape)

plt.imshow(a)
plt.show()