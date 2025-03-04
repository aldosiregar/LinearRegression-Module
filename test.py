import numpy as np
import matplotlib.pyplot as plt
import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

n_samples = 1000
n_features = 1
noise = 10

x, y = make_regression(n_samples=n_samples, n_features=n_features,
                       noise=noise)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=43)

plt.scatter(x=x_train, y=y_train)


lingress = LinearRegression.LinearRegression(x_train=x_train, y_train=y_train, alpha=0.7)

plt.plot(x_train, lingress.predict(x_test=x_train), color="r")

lingress.fit(loss_function_type="RMSE")

print(f"x_test max : {np.max(x_test)}, x_test= {np.min(x_test)}")
print(lingress)

plt.plot(x_train,lingress.predict(x_test=x_train), color="g")
plt.show()