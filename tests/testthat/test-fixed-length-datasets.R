context("fixed length record datasets")


test_succeeds("fixed_length_record_dataset creates a dataset", {

  data_file <-  testing_data_filepath("mtcars-test.csv")
  file_bytes <- file.info(data_file)$size

  dataset <- fixed_length_record_dataset(
    data_file, record_bytes = file_bytes)

  if (tf$executing_eagerly()) {
    batch <- next_batch(dataset)
  } else {
    with_session(function(sess) {
      batch <- next_batch(dataset)
      file_contents <- sess$run(batch)
    })
  }

})
