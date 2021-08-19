context("filenames")


test_succeeds("filenames can be listed", {

  dataset <-
    testing_data_filepath("mtcars*.csv") %>%
    file_list_dataset() %>%
    dataset_interleave(cycle_length = 2, function(file) {
      text_line_dataset(file)
    })
})
