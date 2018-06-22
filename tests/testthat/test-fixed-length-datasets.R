context("fixed length record datasets")

source("utils.R")

test_succeeds("fixed_length_record_dataset creates a dataset", {

  data_file <- "data/mtcars-test.csv"
  file_bytes <- file.info(data_file)$size

  dataset <- fixed_length_record_dataset(
    "data/mtcars-test.csv", record_bytes = file_bytes)

  with_session(function(sess) {
    batch <- next_batch(dataset)
    file_contents <- sess$run(batch)
  })

})











