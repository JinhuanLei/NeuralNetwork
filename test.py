import numpy as np
for i in range(3,0,-1):
    print(i)
x = np.array([1,2,3,4])
print(x.shape)
x = x.reshape(-1,1)
print(x.shape)
for i in range(4):
    print(x[i])