context("sql datasets")

source("utils.R")

test_succeeds("sqlite_dataset creates a dataset", {

  skip_if_no_tensorflow()

  dataset <- sqlite_dataset(
    'data/mtcars.sqlite3',
    'select * from mtcars',
    list(tf$double, tf$int32, tf$double, tf$int32, tf$double, tf$double,
         tf$double, tf$int32, tf$int32, tf$int32, tf$int32)
  )

})











