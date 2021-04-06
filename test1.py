import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(10,1000,10)
x = np.linspace(0,7,num=5000)
y = np.sin(x)
figure1 = plt.plot(x,y)
plt.show()
