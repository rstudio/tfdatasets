context("filenames")

source("utils.R")

test_succeeds("filenames can be listed", {

  dataset <- file_list_dataset("data/mtcars*.csv") %>%
    dataset_interleave(cycle_length = 2, function(file) {
      text_line_dataset(file)
    })
})








