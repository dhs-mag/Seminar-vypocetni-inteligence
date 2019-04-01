import numpy as np


C = np.array([
    [8, 9, -4, 5],
    [3, -8, 3, -2],
    [-4, 2, -7, 1],
    [-8, -5, -7, -6]
])
Y = np.array([-7, 3, 8, -3])
Ci = np.linalg.inv(C)

X = Ci @ Y
YY = C @ X
print(X)
print ("Overeni")
print(YY)