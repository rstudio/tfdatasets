context("text datasets")

source("utils.R")

test_succeeds("text_line_dataset creates a dataset", {
  text_line_dataset("data/mtcars.csv")
})


test_succeeds("text_line_dataset can read gzip datasets", {
  text_line_dataset("data/mtcars.tar.gz", compression_type = "GZIP")
})









