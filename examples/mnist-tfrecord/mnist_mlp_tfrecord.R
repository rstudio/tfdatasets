# Train MNIST with tfrecords yielded from a TF Dataset

# In order to run this example you should first run 'mnist_to_tfrecord.py'
# which will download MNIST data and serialize it into 3 tfrecords files
# (train.tfrecords, validation.tfrecords, and test.tfrecords).
#
# Also note that this example requires:
#   - Keras >= 2.2
#   - TF >= 1.8
#   - The dev version of tfdatasets
#     (devtools::install_github("rstudio/tfdatasets"))
#

library(keras)
library(tfdatasets)

batch_size = 128
steps_per_epoch = 500

# function to read and preprocess mnist dataset
mnist_dataset <- function(filename) {
  dataset <- tfrecord_dataset(filename, num_parallel_reads = 8) %>%
    dataset_shuffle_and_repeat(1000) %>%
    dataset_map_and_batch(batch_size = batch_size, drop_remainder = TRUE,
      function(example_proto) {
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
    dataset_prefetch(1)
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


history <- model %>% fit(
  mnist_dataset("mnist/train.tfrecords"),
  steps_per_epoch = steps_per_epoch,
  epochs = 20,
  validation_data = mnist_dataset("mnist/validation.tfrecords"),
  validation_steps = steps_per_epoch
)


plot(history)

score <- model %>% evaluate(
  mnist_dataset("mnist/test.tfrecords"),
  steps = steps_per_epoch
)

print(score)


