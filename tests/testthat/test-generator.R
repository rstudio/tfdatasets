context("generator")

library(keras)
K <- backend()
K$get_session()

source("utils.R")

test_succeeds("generator_from_dataset feeds data to fit_generator", {

  # create model
  model <- keras_model_sequential() %>%
    layer_dense(units = 10, activation = "relu", input_shape = c(4)) %>%
    layer_dense(units = 20, activation = "relu") %>%
    layer_dense(units = 3, activation = "softmax") %>%
    compile(
      loss = 'categorical_crossentropy',
      optimizer = optimizer_rmsprop(),
      metrics = c('accuracy')
    )

  # create dataset
  dataset <- csv_dataset("data/iris.csv") %>%
    dataset_shuffle(50) %>%
    dataset_batch(10) %>%
    dataset_repeat(5)

  # generator to stream values from the dataset
  iris_generator <- generator_from_dataset(dataset, features = -Species, response = Species)

  # fit with the generator
  model %>% fit_generator(
    generator = iris_generator,
    steps_per_epoch = 15,
    epochs = 5
  )



})











