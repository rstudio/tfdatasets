context("read-files")


test_succeeds("files can be read in parallel", {

  mtcars_spec <- csv_record_spec(testing_data_filepath("mtcars.csv"))

  dataset <- read_files(testing_data_filepath("mtcars-*.csv"), text_line_dataset, record_spec = mtcars_spec,
                        parallel_files = 4, parallel_interleave = 1,
                        num_shards = 2, shard_index = 0) %>%
    dataset_batch(1000)

  if (tf$executing_eagerly()) {
    batch <- next_batch(dataset)
  } else {
    with_session(function(sess) {
      batch <- next_batch(dataset)
      data <- sess$run(batch)
    })
  }

})
