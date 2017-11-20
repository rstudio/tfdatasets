'''Train MNIST with tfrecords yielded from a TF Dataset

In order to run this example you should first run 'mnist_to_tfrecord.py'
which will download MNIST data and serialize it into 3 tfrecords files
(train.tfrecords, validation.tfrecords, and test.tfrecords).

This example demonstrates the use of TF Datasets wrapped by a generator
function. The example currently only works with a fork of keras that accepts
`workers=0` as an argument to fit_generator, etc. Passing `workers=0` results
in the generator function being run on the main thread (without this various
errors ensue b/c of the way TF handles being called on a background thread).
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import keras.backend as K

import tensorflow as tf

batch_size = 128
num_classes = 10
epochs = 20
steps_per_epoch = 500

# Return a TF dataset for specified filename(s)
def mnist_dataset(filenames):

  def decode_example(example_proto):

    features = tf.parse_single_example(
      example_proto,
      features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
      }
    )

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32) / 255.

    label = tf.one_hot(tf.cast(features['label'], tf.int32), num_classes)

    return [image, label]

  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(decode_example)
  dataset = dataset.repeat()
  dataset = dataset.shuffle(10000)
  dataset = dataset.batch(batch_size)
  return dataset


# Keras generator that yields batches from the speicfied tfrecord filename(s)
def mnist_generator(filenames):

  dataset = mnist_dataset(filenames)
  iter = dataset.make_one_shot_iterator()
  batch = iter.get_next()

  while True:
    yield K.batch_get_value(batch)


model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(784,)))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit_generator(
  mnist_generator('mnist/train.tfrecords'),
  steps_per_epoch=steps_per_epoch,
  epochs=epochs,
  verbose=1,
  validation_data=mnist_generator('mnist/validation.tfrecords'),
  validation_steps=steps_per_epoch,
  workers = 0  # runs generator on the main thread
)

score = model.evaluate_generator(
  mnist_generator('mnist/test.tfrecords'),
  steps=steps_per_epoch,
  workers = 0  # runs generator on the main thread
)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


