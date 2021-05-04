from LinearRegression import LinearRegression as lin_reg
import numpy as np
import matplotlib.pyplot as plt


x = np.random.rand(100,1)
y = 5+10*x + np.random.randn(100,1)


linear_1 = lin_reg(x,y)

linear_1.fit()

linear_1.predict(x)

y_pred = linear_1.y_pred

plt.plot(x,y,'*',x,y_pred,'r--')

plt.show()
