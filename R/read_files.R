


#' Read a dataset from a set of files
#'
#' Read files into a dataset, optionally processing them in parallel.
#'
#' @param files List of filenames or glob pattern for files (e.g. "*.csv")
#' @param reader Function that maps a file into a dataset (e.g.
#'   [text_line_dataset()] or [tfrecord_dataset()]).
#' @param ... Additional arguments to pass to `reader` function
#' @param parallel_files An integer, number of files to process in parallel
#' @param parallel_interleave An integer, number of consecutive records to
#'   produce from each file before cycling to another file.
#' @param num_shards An integer representing the number of shards operating in
#'   parallel.
#' @param shard_index An integer, representing the worker index. Shared indexes
#'  are 0 based so for e.g. 8 shards valid indexes would be 0-7.
#'
#' @return A dataset
#'
#' @export
read_files <- function(files, reader, ...,
                       parallel_files = 1, parallel_interleave = 1,
                       num_shards = NULL, shard_index = NULL) {

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
  dataset <- files_dataset

  # shard
  if (!is.null(num_shards)) {
    dataset <- dataset %>%
      dataset_shard(num_shards = num_shards, index = shard_index)
  }

  # parallel files with interleave
  dataset <- dataset %>%
    dataset_interleave(cycle_length = parallel_files, block_length = parallel_interleave,
                       function(file) {
                         reader(file, ...)
                        })

  # return
  as_tf_dataset(dataset)
}



