# Train MNIST with tfrecords yielded from a TF Dataset

# In order to run this example you should first run 'mnist_to_tfrecord.py'
# which will download MNIST data and serialize it into 3 tfrecords files
# (train.tfrecords, validation.tfrecords, and test.tfrecords).
#

library(keras)
library(tfdatasets)

batch_size = 128
epochs = 20
steps_per_epoch = 500

# function to read and preprocess mnist dataset
mnist_dataset <- function(filename) {
  dataset <- tfrecord_dataset(filename) %>%
    dataset_map(function(example_proto) {

      # parse record
      features <- tf$parse_single_example(
        example_proto,
        features = list(
          image_raw = tf$FixedLenFeature(shape(), tf$string),
          label = tf$FixedLenFeature(shape(), tf$int64)
        )
      )

      # preprocess image
      image <- tf$decode_raw(features$image_raw, tf$uint8)
      image <- tf$cast(image, tf$float32) / 255

      # convert label to one-hot
      label <- tf$one_hot(tf$cast(features$label, tf$int32), 10L)

      # return
      list(image, label)
    }) %>%
    dataset_repeat() %>%
    dataset_shuffle(10000) %>%
    dataset_batch(batch_size)
}

model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit_generator(
  generator = mnist_dataset("mnist/train.tfrecords"),
  steps_per_epoch = steps_per_epoch,
  epochs = epochs,
  validation_data = mnist_dataset("mnist/validation.tfrecords"),
  validation_steps = steps_per_epoch
)

plot(history)

score <- model %>% evaluate_generator(
  generator = mnist_dataset("mnist/test.tfrecords"),
  steps = steps_per_epoch
)

print(score)


