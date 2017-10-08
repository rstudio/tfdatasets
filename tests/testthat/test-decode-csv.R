context("decode_csv")

source("utils.R")

cols <- c("disp", "drat", "vs", "gear", "mpg", "qsec", "hp", "am", "wt",  "carb", "cyl")

test_that("decode_csv parses column names from file", {
  skip_if_no_tensorflow()
  dataset <- csv_dataset("data/mtcars.csv")
  expect_identical(cols, names(dataset$output_shapes))
})

test_that("decode_csv parses explicitly named columns", {
  skip_if_no_tensorflow()
  dataset <- csv_dataset("data/mtcars.csv", col_names = cols) %>%
    dataset_skip(1)
  expect_identical(cols, names(dataset$output_shapes))
})




