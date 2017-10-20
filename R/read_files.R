


#' Read a dataset from a set of files
#'
#' @param files List of filenames or glob pattern for files (e.g. "*.csv")
#' @param reader Function that maps a file into a dataset (e.g.
#'   [csv_dataset()]).
#' @param ... Additional arguments to pass to `reader` function
#' @param num_shards An integer representing the number of shards operating in
#'   parallel.
#' @param shard_index An integer, representing the worker index.
#' @param parallel_files An integer, number of files to process in parallel
#' @param parallel_interleave An integer, number of consecutive records to
#'   produce from each file before cycling to another file.
#'
#' @return A dataset
#'
#' @export
read_files <- function(files, reader, ...,
                       num_shards = 1, shard_index = 1,
                       parallel_files = 1, parallel_interleave = 1) {

  # files dataset to process
  files_dataset <- NULL

  # if files is a character vector
  if (is.character(files)) {

    # if it's a glob pattern then resolve it
    if (length(files) == 1 && !identical(utils::glob2rx(files), files))
      files_dataset <- file_list_dataset(files)
    # otherwise just convert to tensor dataset
    else
      files_dataset <- tensor_slices_dataset(as.list(files))

  # convert tensors to dataset
  } else if (is_tensor(files)) {

    files_dataset <- tensor_slices_dataset(files)

  # already a dataset!
  } else if (is_dataset(files)) {

    files_dataset <- files

  # no idea
  } else {

    stop("Invalid type (", class(files)[[1]], ") for files argument")

  }

  # read with appropriate parallel options
  files_dataset %>%
    dataset_shard(num_shards = num_shards, index = shard_index) %>%
    dataset_interleave(cycle_length = parallel_files, block_length = parallel_interleave,
                       function(file) { reader(file, ...) })
}



