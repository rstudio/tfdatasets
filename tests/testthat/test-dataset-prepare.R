context("dataset-prepare")

source("utils.R")

require(keras)

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
  batch <- mtcars_dataset_nobatch() %>%
    dataset_prepare(cyl ~ mpg + disp, batch_size = 16L) %>%
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

test_succeeds("dataset_prepare can use names from a string", {
  batch <- mtcars_dataset() %>%
    dataset_prepare(x = colnames(mtcars)[c(1,3)], named = TRUE) %>%
    next_batch()

  expect_length(setdiff(names(batch), c("x")), 0)
})

test_succeeds("dataset_prepare can use names from unquoting", {

  x <- c("disp", "mpg")

  batch <- mtcars_dataset() %>%
    dataset_prepare(x = !!x, named = TRUE) %>%
    next_batch()

  expect_length(setdiff(names(batch), c("x")), 0)
})

test_succeeds("dataset_prepare can provide keras input tensors", {

  if (tensorflow::tf_version() < "1.13")
    skip("dataset_prepare required TF >= 1.13")

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

  if (tensorflow::tf_version() >= "1.14" && tensorflow::tf$executing_eagerly()) {

    # You should not pass an EagerTensor to `Input`.
    # For example, instead of creating an InputLayer, you should instantiate your model and directly
    # call it on your input.
    model <- tf$keras$Sequential(
      list(tf$keras$layers$Dense(10L, tf$nn$relu),
           tf$keras$layers$Dense(10L, tf$nn$relu),
           tf$keras$layers$Dense(3L, tf$nn$softmax))
    )

    model$compile(loss = "categorical_crossentropy",
                  optimizer = "adam",
                  metrics = list("accuracy"))

    model$fit(train_batch$x,
              train_batch$y,
              epochs = 5L)

  } else {

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

    # see above
    model %>% fit(
      steps_per_epoch = 15L,
      epochs = 5
    )
  }

})
