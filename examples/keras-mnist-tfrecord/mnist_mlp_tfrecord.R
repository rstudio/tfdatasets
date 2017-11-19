
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
        )
      )

      # preprocess image
      image <- tf$decode_raw(features$image_raw, tf$uint8)
      image <- tf$cast(image, tf$float32) / 255

      # convert label to integer
      label <- tf$cast(features$label, tf$int32)

      # return
      list(
        image = image,
        label = label
      )
    }) %>%
    dataset_batch(128) %>%
    dataset_repeat()
}

train_dataset <- mnist_dataset("mnist/train.tfrecords")

