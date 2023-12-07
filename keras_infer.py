import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from math import floor
path = 'keras/model64_gen_LATEST'
epoch = 29

path = f"keras/64run_{floor(epoch/10)}_20epoch/model64_gen_{epoch % 10}"
print(path)
model = tf.keras.models.load_model(path)
prediction = (model.predict(tf.random.normal(shape=(1,128))))
#prediction *= 255
prediction = prediction[0]


fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
l = ax.imshow(prediction)

model.summary()
class Regen:
    def refresh(self, event):
        prediction = (model.predict(tf.random.normal(shape=(1,128))))[0]
        ax.imshow(prediction)
        plt.draw()


callback = Regen()
axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.refresh)
plt.show()
#print(tf.random.normal(shape=(1,128)))

