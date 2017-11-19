
library(tfdatasets)

# function to read and preprocess mnist dataset
mnist_dataset <- function(filename) {
  dataset <- tfrecord_dataset(filename) %>%
    dataset_map(function(example_proto) {
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

      # convert label to one-hot integer
      label <- tf$one_hot(tf$cast(features$label, tf$int32), 10L)

      # return
      list(image, label)
    }) %>%
    dataset_batch(128) %>%
    dataset_repeat()
}

train_dataset <- mnist_dataset("mnist/train.tfrecords")
validation_dataset <- mnist_dataset("mnist/validation.tfrecords")
test_dataset <- mnist_dataset("mnist/test.tfrecords")
