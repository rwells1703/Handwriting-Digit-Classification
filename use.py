import struct
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("num_read.model")

with open("test.bmp", "rb") as f:
    test_image_bytes = f.read()

test_image_bytes_no_header = test_image_bytes[54:]
test_image = list(struct.iter_unpack("BBB", test_image_bytes_no_header))

test_image_greyscale = []
i = 0
while i < len(test_image):
    pixel = test_image[i]

    if i % 28 == 0:
        test_image_greyscale.insert(0, [])

    test_image_greyscale[0].append(255-(pixel[0] + pixel[1] + pixel[1])//3)

    i += 1

predictions = model.predict([test_image_greyscale])

print(np.argmax(predictions[0]))
plt.imshow(test_image_greyscale, cmap = plt.cm.binary)
plt.show()