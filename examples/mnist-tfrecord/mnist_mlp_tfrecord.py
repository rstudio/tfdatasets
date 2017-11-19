

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


def mnist_generator(filenames):

  dataset = mnist_dataset(filenames)
  iter = dataset.make_one_shot_iterator()
  batch = iter.get_next()

  while True:
    yield K.batch_get_value(batch)


sess = tf.Session()
with sess.as_default():

  train_generator = mnist_generator('mnist/train.tfrecords')
  validation_generator = mnist_generator('mnist/validation.tfrecords')
  test_generator =  mnist_generator('mnist/test.tfrecords')

  model = Sequential()
  model.add(Dense(512, activation='relu', input_shape=(784,)))
  model.add(Dropout(0.2))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(num_classes, activation='softmax'))

  model.summary()

  model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])



  history = model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch
  )

  score = model.evaluate_generator(
    test_generator,
    steps=steps_per_epoch
  )

  print('Test loss:', score[0])
  print('Test accuracy:', score[1])


