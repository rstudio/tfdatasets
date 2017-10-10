context("input_fn")

library(tfestimators)

source("utils.R")

use_input_fn <- function(features, response) {

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

# test_succeeds("input_fn supports NULL response", {
#   use_input_fn(features = c("disp", "cyl"), response = NULL)
# })

test_succeeds("input_fn supports tidyselect", {

  dataset <- csv_dataset("data/mtcars-train.csv") %>%
    dataset_shuffle(20) %>%
    dataset_batch(10) %>%
    dataset_repeat(5)

  # create input_fn from dataset
  input_fn(dataset, features = c(disp, cyl), response = mpg)
})








