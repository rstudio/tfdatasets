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
  dataset <- csv_dataset("data/mtcars.csv", col_names = cols) %>%
    dataset_skip(1)
  expect_length(setdiff(cols, names(dataset$output_shapes)), 0)
})

test_succeeds("decode_csv handles explicit record default/type specifications", {
  skip_if_no_tensorflow()
  csv_dataset("data/mtcars.csv", record_defaults = list(0,0L,0,0L,0,0,0,0L,0L,0L,0L))
  csv_dataset("data/mtcars.csv", record_defaults = list(0,0,0,0,0,0,0,0,0,0,0))
})




