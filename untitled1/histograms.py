import matplotlib.pyplot as plt
import numpy as np

values = np.random.normal(25,3,1000)
plt.hist(values,bins=10)
plt.show()
plt.clf()

values_2 = values * 0.65
plt.hist(values_2,bins=20)
plt.show()
plt.clf()