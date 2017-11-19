
library(keras)
library(tfdatasets)

mnist_dataset <- function(filename) {
  dataset <- tfrecord_dataset(filename) %>%
    dataset_map(function(example_proto) {
      features <- tf$parse_single_example(
        example_proto,
        features = list(
          image_raw = tf$FixedLenFeature(shape(), tf$string),
          label = tf$FixedLenFeature(shape(), tf$int64)
        ))
      list(
        image = tf$reshape(
          tf$decode_raw(features$image_raw, tf$uint8),
          shape(28, 28)
        ),
        label = tf$cast(features$label, tf$int32)
      )
    })
  # %>%
  #   dataset_batch(128) %>%
  #   dataset_repeat()
}

train_dataset <- mnist_dataset("mnist/train.tfrecords")


dataset_map(function(example_proto) {
  features <- list(
    image = tf$FixedLenFeature(shape(), tf$string, default_value = ""),
    label = tf$FixedLenFeature(shape(), tf$int32, default_value = 0L)
  )
  tf$parse_single_example(example_proto, features)
})
