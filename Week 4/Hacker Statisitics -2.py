import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

np.random.seed(42)
random_numbers = np.random.random(size=4)
print(random_numbers)

total_4heads = 0

for i in range (10000):
    heads = np.random.random(4)< 0.5
    n_heads = np.sum(heads)
    if n_heads ==4:
        total_4heads +=1

print(total_4heads/10000)

# Another way to generate random number of heads.
# np.random.binomial ( number of flips per game, probability of getting a head, number of games played)

x = np.random.binomial(100, 0.5,size= 10000)
plt.hist(x, bins= 100)
plt.show()

