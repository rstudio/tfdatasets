context("filenames")

source("utils.R")

test_succeeds("filenames can be specified via globbing", {
  text_line_dataset("data/mtcars*.csv")
})

test_succeeds("multiple filename glob patterns are handled", {
  text_line_dataset(c("data/mtcars.csv", "data/mtcars-*.csv"))
})









