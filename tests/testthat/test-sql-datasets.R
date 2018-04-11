context("sql datasets")

source("utils.R")

test_succeeds("sqlite_dataset creates a dataset", {

  skip_if_no_tensorflow()

  record_spec <- sql_record_spec(
    names = c("disp", "drat", "vs", "gear", "mpg", "qsec", "hp", "am", "wt",  "carb", "cyl"),
    types = c(tf$float64, tf$int32, tf$float64, tf$int32, tf$float64, tf$float64,
              tf$float64, tf$int32, tf$int32, tf$int32, tf$int32)
  )

  dataset <- sqlite_dataset(
    'data/mtcars.sqlite3',
    'select * from mtcars',
    record_spec
  )

})











