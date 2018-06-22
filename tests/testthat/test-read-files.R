context("read-files")

source("utils.R")

test_succeeds("files can be read in parallel", {

  mtcars_spec <- csv_record_spec("data/mtcars.csv")

  dataset <- read_files("data/mtcars-*.csv", text_line_dataset, record_spec = mtcars_spec,
                        parallel_files = 4, parallel_interleave = 1,
                        num_shards = 2, shard_index = 0) %>%
    dataset_batch(1000)

  with_session(function(sess) {
    batch <- next_batch(dataset)
    data <- sess$run(batch)
  })


})








