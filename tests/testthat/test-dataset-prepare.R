context("dataset-prepare")

source("utils.R")

test_succeeds("dataset_prepare yields list of x and y tensors", {
  batch <- mtcars_dataset() %>%
    dataset_prepare(x = c(mpg, disp), y = cyl) %>%
    next_batch()
  expect_length(setdiff(names(batch), c("x", "y")), 0)
})

test_succeeds("dataset_prepare accepts formula syntax", {
  batch <- mtcars_dataset() %>%
    dataset_prepare(cyl ~ mpg + disp) %>%
    next_batch()
  expect_length(setdiff(names(batch), c("x", "y")), 0)
})

test_succeeds("dataset_prepare can fuse dataset_batch", {
  batch <- mtcars_dataset() %>%
    dataset_prepare(cyl ~ mpg + disp, batch_size = 16) %>%
    next_batch()
  expect_length(setdiff(names(batch), c("x", "y")), 0)
})

test_succeeds("dataset_prepare does not require y", {
  batch <- mtcars_dataset() %>%
    dataset_prepare(x = c(mpg, disp)) %>%
    next_batch()
  expect_null(batch$y)
})

test_succeeds("dataset_prepare can return named features", {
  batch <- mtcars_dataset() %>%
    dataset_prepare(x = c(mpg, disp), named_features = TRUE) %>%
    next_batch()
  expect_length(setdiff(names(batch$x), c("mpg", "disp")), 0)
})

test_succeeds("dataset_prepare can return an unnamed list", {
  batch <- mtcars_dataset() %>%
    dataset_prepare(x = c(mpg, disp), named = FALSE) %>%
    next_batch()
  expect_null(names(batch))
})

test_succeeds("dataset_prepare can provide keras input tensors", {

  require(keras)

  # create dataset
  dataset <- csv_dataset("data/iris.csv") %>%
    dataset_map(function(record) {
      record$Species <- tf$one_hot(record$Species, depth = 3L)
      record
    }) %>%
    dataset_prepare(x = -Species, y = Species) %>%
    dataset_shuffle(50) %>%
    dataset_batch(10) %>%
    dataset_repeat() # repeat infinitely

  # stream batches from dataset
  train_batch <- next_batch(dataset)

  # create model
  input <- layer_input(tensor = train_batch$x, shape = c(4))
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
