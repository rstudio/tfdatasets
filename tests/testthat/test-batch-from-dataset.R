context("batch-from-dataset")


source("utils.R")

mtcars_dataset <- function() {
  csv_dataset("data/mtcars.csv") %>%
    dataset_shuffle(50) %>%
    dataset_batch(10)
}

test_succeeds("batch_from_dataset yields list of x and y tensors", {
  batch <- batch_from_dataset(mtcars_dataset(), features = c(mpg, disp), response = cyl)
  expect_named(batch, c("x", "y"))
})

test_succeeds("batch_from_dataset does not require response", {
  batch <- batch_from_dataset(mtcars_dataset(), features = c(mpg, disp))
  expect_null(batch$y)
})

test_succeeds("batch_from_dataset can return named features", {
  batch <- batch_from_dataset(mtcars_dataset(), features = c(mpg, disp), named_features = TRUE)
  expect_named(batch$x, c("mpg", "disp"))
})

test_succeeds("batch_from_dataset can return an unnamed list", {
  batch <- batch_from_dataset(mtcars_dataset(), features = c(mpg, disp), named = FALSE)
  expect_null(names(batch))
})

test_succeeds("batch_from_dataset can provide keras input tensors", {

  library(keras)
  K <- backend()
  K$get_session()

  # create dataset
  dataset <- csv_dataset("data/iris.csv") %>%
    dataset_map(function(record) {
      record$Species <- tf$one_hot(record$Species, depth = 3L)
      record
    }) %>%
    dataset_shuffle(50) %>%
    dataset_batch(10) %>%
    dataset_repeat() # repeat infinitely

  # stream batches from dataset
  train_batch <- batch_from_dataset(dataset, features = -Species, response = Species)

  # create model
  input <- layer_input(tensor = train_batch$x, batch_shape = shape(NULL, 4))
  predictions <- input %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dense(units = 20, activation = "relu") %>%
    layer_dense(units = 3, activation = "softmax")
  model <- keras_model(input, predictions)
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy'),
    target_tensors = train_batch$y
  )

  # fit with the generator
  model %>% fit(
    steps_per_epoch = 15L,
    epochs = 5
  )
})
