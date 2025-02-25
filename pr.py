import numpy as np
from sklearn.linear_model import LinearRegression

size = np.array([[500], [700], [900]])
rent = np.array([1000, 1500, 2000])

model = LinearRegression()
model.fit(size, rent)

new_size = np.array([[1300]])
pred = model.predict(new_size)

print('Predicted rent for 1300 sq ft:', pred[0])