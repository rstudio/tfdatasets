context("decode_csv")

source("utils.R")

cols <- c("disp", "drat", "vs", "gear", "mpg", "qsec", "hp", "am", "wt",  "carb", "cyl")

test_that("decode_csv parses column names from file", {
  skip_if_no_tensorflow()
  dataset <- csv_dataset("data/mtcars.csv")
  expect_length(setdiff(cols, names(dataset$output_shapes)), 0)
})

test_that("decode_csv parses explicitly named columns", {
  skip_if_no_tensorflow()
  dataset <- csv_dataset("data/mtcars.csv", names = cols) %>%
    dataset_skip(1)
  expect_length(setdiff(cols, names(dataset$output_shapes)), 0)
})

test_that("decode_csv validates type specififers", {
  skip_if_no_tensorflow()
  expect_error(csv_dataset("data/mtcars.csv", types = c("logical", "numeric", "double", "integer", "double", "double", "double", "integer", "integer", "integer", "integer")))
  expect_error(csv_dataset("data/mtcars.csv", types = c("witv")))
})

test_that("decode_csv rejects name/type/default specifiers of the wrong length", {
  skip_if_no_tensorflow()
  expect_error(csv_dataset("data/mtcars.csv", names = c("foo", "bar")))
  expect_error(csv_dataset("data/mtcars.csv", types = c("integer", "double")))
  expect_error(csv_dataset("data/mtcars.csv", types = c("id")))
  expect_error(csv_dataset("data/mtcars.csv", defaults = list(0L, 0)))
})

test_succeeds("decode_csv handles column type specifications", {
  csv_dataset("data/mtcars.csv", types = c("double", "integer", "double", "integer", "double", "double", "double", "integer", "integer", "integer", "integer"))
  csv_dataset("data/mtcars.csv", types = rep_len("double", 11))
})

test_succeeds("decode_csv handles column type abbreviations", {
  csv_dataset("data/mtcars.csv", types = "dididddiiii")
  csv_dataset("data/mtcars.csv", types = "ddddddddddd")
})

test_succeeds("decode_csv handles explicit record defaults", {
  csv_dataset("data/mtcars.csv", types = "ddddddddddd", defaults = list(0,0,0,0,0,0,0,0,0,0,0))
})

test_succeeds("decode_csv can impute types from defaults", {
  csv_dataset("data/mtcars.csv", defaults = list(0,0,0,0,0,0,0,0,0,0,0))
})

test_succeeds("decode_csv does not attempt a preview if col info is provided", {
  csv_dataset("data/foo.csv", names = c("a", "b", "c"), defaults = list(0,0,0))
})

test_succeeds("make_csv_dataset can read a dataset", {
  make_csv_dataset("data/mtcars-*.csv", batch_size = 10)
})




