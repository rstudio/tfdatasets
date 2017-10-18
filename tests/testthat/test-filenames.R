context("filenames")

source("utils.R")

test_succeeds("filenames can be listed", {
  file_list_dataset("data/mtcars*.csv")
})











