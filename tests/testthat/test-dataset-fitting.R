context("dataset-fitting")

source("utils.R")

library(keras)

test_succeeds("can fit a dataset using keras fit", {
  d  <-
    tensor_slices_dataset(list(matrix(1:20, ncol = 2), matrix(1:10, ncol = 1))) %>%
    dataset_batch(2) %>%
    dataset_shuffle(1024)

  model <- keras_model_sequential() %>%
    layer_dense(units = 1, input_shape = 2)

  model %>% compile(optimizer = "adam", loss = "mse")
  model %>% fit(d, steps_per_epoch = 5, epochs = 1)
})

test_succeeds("can fit a dataset using keras fit_generator", {
  d  <-
    tensor_slices_dataset(list(matrix(1:20, ncol = 2), matrix(1:10, ncol = 1))) %>%
    dataset_batch(2) %>%
    dataset_shuffle(1024)


  model <- keras_model_sequential() %>%
    layer_dense(units = 1, input_shape = 2)

  model %>% compile(optimizer = "adam", loss = "mse")
  model %>% fit_generator(d, steps_per_epoch = 5, epochs = 1)
})

