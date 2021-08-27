context("text datasets")

test_succeeds("text_line_dataset creates a dataset", {
  text_line_dataset(testing_data_filepath("mtcars.csv"))
})


test_succeeds("text_line_dataset can read gzip datasets", {
  text_line_dataset(testing_data_filepath("mtcars.tar.gz"), compression_type = "GZIP")
})
