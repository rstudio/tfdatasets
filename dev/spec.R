

# generic record spec (used e.g. for binary tfrecord format, which has no
# equivalent of delim or skip, and can't be previewed)
record_spec <- function(names = NULL, types = NULL, defaults = NULL) {
  list(
    names = names,
    types = types,       # types and defaults can be inferred from eachother
    defaults = defaults
  )
}

# spec for delimited files (note: requires skip for both padding and for when
# names is given explicitly, as in that case we need to know whether the first
# line of the file has names or not)
delim_record_spec <- function(example_file, delim, skip = 0,
                              names = NULL, types = NULL, defaults = NULL) {

}

# specialization for CSV
csv_record_spec <- function(example_file, skip = 0,
                            names = NULL, types = NULL, defaults = NULL) {

}

# create a record spec with help from an example
record_spec <- csv_record_spec("example.csv", col_types = "ddddii")

# read in a single file
dataset <- csv_dataset("mtcars.csv", record_spec = record_spec)

# read in multiple files
dataset <- file_list_dataset("*.csv") %>%
  dataset_flat_map(function(file) {
    csv_dataset(file, record_spec = record_spec)
  })

# read 4 files in parallel
dataset <- file_list_dataset("*.csv") %>%
  dataset_interleave(cycle_length = 4, function(file) {
    csv_dataset(file, record_spec = record_spec)
  })

# read sharded set of files in parallel
dataset <- file_list_dataset("*.csv") %>%
  dataset_shard(num_shards = 10, index = 1) %>%
  dataset_interleave(cycle_length = 4, function(file) {
    csv_dataset(file, record_spec = record_spec)
  })






