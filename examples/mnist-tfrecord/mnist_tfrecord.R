
library(keras)
library(tfdatasets)


# see keras flow_images_from_directory
dataset_prepare_images <- function(dataset,
                                   target_size = c(256, 256), color_mode = "rgb",
                                   classes = NULL, class_mode = "categorical") {

}


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
    dataset_batch(128)
}

train_dataset <- mnist_dataset("mnist/train.tfrecords")
validation_dataset <- mnist_dataset("mnist/validation.tfrecords")
test_dataset <- mnist_dataset("mnist/test.tfrecords")


summary(model)

data_mode <- "generator"

if (data_mode == "generator") {

  model <- keras_model_sequential()
  model %>%
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
    generator = train_dataset,
    steps_per_epoch = 500,
    epochs = 30,
    validation_data = validation_dataset,
    validation_steps = 40,
    verbose = 1
  )

  results <- model %>% evaluate_generator(
    generator = test_dataset,
    steps = 80
  )

} else {

  batch <- next_batch(train_dataset)
  x_train_batch <- batch[[1]]
  y_train_batch <- batch[[2]]

  inputs <- layer_input(tensor = x_train_batch, batch_shape = list(NULL, 784))

  outputs <- inputs %>%
    layer_dense(units = 256, activation = 'relu') %>%
    layer_dropout(rate = 0.4) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 10, activation = 'softmax')

  model <- keras_model(inputs, outputs)

  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy'),
    target_tensors = y_train_batch
  )

  history <- model %>% fit(
    steps_per_epoch = 500,
    epochs = 30,
  )
}






