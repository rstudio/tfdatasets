context("input_fn")

source("utils.R")

use_input_fn <- function(features, response) {

  require(tfestimators)

  # return an input_fn for a set of csv files
  mtcars_input_fn <- function(filenames) {

    # dataset w/ batch size of 10 that repeats for 5 epochs
    dataset <- csv_dataset(filenames) %>%
      dataset_shuffle(20) %>%
      dataset_batch(10) %>%
      dataset_repeat(5)

    # create input_fn from dataset
    input_fn(dataset, features, response)
  }

  # define feature columns
  cols <- feature_columns(
    column_numeric("disp"),
    column_numeric("cyl")
  )

  # create model
  model <- linear_regressor(feature_columns = cols)

  # train model
  model %>% train(mtcars_input_fn("data/mtcars-train.csv"))

  # evaluate model
  model %>% evaluate(mtcars_input_fn("data/mtcars-test.csv"))
}

test_succeeds("input_fn feeds data to train and evaluate", {
  use_input_fn(features = c("disp", "cyl"), response = "mpg")
})

test_that("input_fn reports incorrect features", {
  skip_if_no_tensorflow()
  expect_error(
    use_input_fn(features = c("displacement", "cylinder"), response = "mpg")
  )
})

test_that("input_fn reports incorrect response", {
  skip_if_no_tensorflow()
  expect_error(
    use_input_fn(features = c("disp", "cyl"), response = "m_p_g")
  )
})

test_that("input_fn rejects un-named datasets", {
  skip_if_no_tensorflow()
  expect_error({
    dataset <- tensors_dataset(1:100)
    input_fn(dataset, features = c("disp", "cyl"), response = "mpg")
  })
})


test_succeeds("input_fn supports tidyselect", {

  dataset <- csv_dataset("data/mtcars-train.csv") %>%
    dataset_shuffle(2000) %>%
    dataset_batch(128) %>%
    dataset_repeat(3)

  # create input_fn from dataset
  input_fn(dataset, features = c(disp, cyl), response = mpg)
})


test_succeeds("input_fn accepts formula syntax", {

  dataset <- csv_dataset("data/mtcars-train.csv") %>%
    dataset_shuffle(2000) %>%
    dataset_batch(128) %>%
    dataset_repeat(3)

  # create input_fn from dataset
  input_fn(dataset, mpg ~ disp + cyl)
})


test_succeeds("input_fn works with custom estimators", {

  skip_if_no_tensorflow()

  require(tfestimators)

  # define custom estimator model_fn
  simple_custom_model_fn <- function(features, labels, mode, params, config) {

    # Create three fully connected layers respectively of size 10, 20, and 10 with
    # each layer having a dropout probability of 0.1.
    logits <- features %>%
      tf$contrib$layers$stack(
        tf$contrib$layers$fully_connected, c(10L, 20L, 10L),
        normalizer_fn = tf$contrib$layers$dropout,
        normalizer_params = list(keep_prob = 0.9)) %>%
      tf$contrib$layers$fully_connected(3L, activation_fn = NULL) # Compute logits (1 per class) and compute loss.

    predictions <- list(
      class = tf$argmax(logits, 1L),
      prob = tf$nn$softmax(logits))

    if (mode == "infer") {
      return(estimator_spec(mode = mode, predictions = predictions, loss = NULL, train_op = NULL))
    }

    labels <- tf$one_hot(labels, 3L)
    loss <- tf$losses$softmax_cross_entropy(labels, logits)

    # Create a tensor for training op.
    train_op <- tf$contrib$layers$optimize_loss(
      loss,
      tf$contrib$framework$get_global_step(),
      optimizer = 'Adagrad',
      learning_rate = 0.1)

    return(estimator_spec(mode = mode, predictions = predictions, loss = loss, train_op = train_op))
  }


  # define dataset
  col_names <- c("SepalLength", "SepalWidth", "PetalLength", "PetalWidth","Species")
  dataset <- csv_dataset("data/iris.csv", names = col_names, types = "ddddi", skip = 1) %>%
    dataset_map(function(record) {
      record$Species <- tf$cast(record$Species, tf$int32)
      record
    }) %>%
    dataset_shuffle(20) %>%
    dataset_batch(10) %>%
    dataset_repeat(5)

  # create model
  classifier <- estimator(model_fn = simple_custom_model_fn, model_dir = tempfile())

  # train
  train(classifier, input_fn(dataset, features = -Species, response = Species))


})







