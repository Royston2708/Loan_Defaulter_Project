import matplotlib.pyplot as plt
import numpy as np

values = np.random.exponential(100,1000)
plt.hist(values,bins=10)
plt.xlabel("Values")
plt.ylabel("Magnitude")
plt.title("Test Plot of Random Numbers")
plt.yticks([0,50,100,150,200,250])

plt.show()
plt.clf()

values_2 = values * 0.65
plt.hist(values_2,bins=20)
plt.show()
plt.clf()


# Week 3 continued
plt.scatter(values)