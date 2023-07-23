import numpy as np
import pandas as pd 
import tensorflow as tf
from matplotlib import pyplot as plt


x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype = float)
y = np.array([3.0, 5.0, 7.0, 9.0, 11.0],dtype = float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 1 , input_shape = [1]),
    tf.keras.layers.Dense(units = 2),
    tf.keras.layers.Dense(units = 3),
    tf.keras.layers.Dense(units = 4),
    tf.keras.layers.Dense(units = 5)
    ])

model.summary()
model.compile(optimizer = "sgd" , loss = "mean_squared_error")

model.fit(x , y, epochs = 500)
# Store the training history
history = model.fit(x, y, epochs=500)

# Access the loss values from the history
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
#PLot the loss function
plt.title("The Convergence Of The Loss Function Against Epochs")
plt.xlabel("epochs")
plt.ylabel("Mean Squared Error")
plt.plot(epochs,loss,label="loss")
plt.legend()
plt.grid(True)
plt.show()


print(model.predict([100]))


