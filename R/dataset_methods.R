

#' Repeats a dataset count times.
#'
#' @param dataset A dataset
#' @param count (Optional.) An integer value representing the number of times
#'   the elements of this dataset should be repeated. The default behavior (if
#'   `count` is `NULL` or `-1`) is for the elements to be repeated indefinitely.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_repeat <- function(dataset, count = NULL) {
  dataset$`repeat`(
    count = as_integer_tensor(count)
  )
}


#' Randomly shuffles the elements of this dataset.
#'
#' @param dataset A dataset
#'
#' @param buffer_size An integer, representing the number of elements from this
#'   dataset from which the new dataset will sample.
#' @param seed (Optional) An integer, representing the random seed that will be
#'   used to create the distribution.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_shuffle <- function(dataset, buffer_size, seed = NULL) {
  dataset$shuffle(
    buffer_size = as_integer_tensor(buffer_size),
    seed = as_integer_tensor(seed)
  )
}

#' Combines consecutive elements of this dataset into batches.
#'
#' @param dataset A dataset
#' @param batch_size An integer, representing the number of consecutive elements
#'   of this dataset to combine in a single batch.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_batch <- function(dataset, batch_size) {
  dataset$batch(
    batch_size = as_integer_tensor(batch_size)
  )
}


#' Splits elements of this dataset into sequences of consecutive elements.
#'
#' For example, if elements of this dataset are shaped `[B, a0, a1, ...]`, where B
#' may vary from element to element, then for each element in this dataset, the
#' unbatched dataset will contain B consecutive elements of shape `[a0, a1, ...]`.
#'
#' @param dataset A dataset
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_unbatch <- function(dataset) {
  dataset$unbatch()
}


#' Caches the elements in this dataset.
#'
#'
#' @param dataset A dataset
#' @param filename String with the name of a directory on the filesystem to use
#'   for caching tensors in this Dataset. If a filename is not provided, the
#'   dataset will be cached in memory.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_cache <- function(dataset, filename = NULL) {
  if (is.null(filename))
    filename <- ""
  if (!is.character(filename))
    stop("filename must be a character vector")
  dataset$cache(tf$constant(filename, dtype = tf$string))
}


#' Creates a dataset by concatenating given dataset with this dataset.
#'
#' @note Input dataset and dataset to be concatenated should have same nested
#'   structures and output types.
#'
#' @param dataset A dataset
#' @param other Dataset to be concatenated
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_concatenate <- function(dataset, other) {
  dataset$concatenate(other)
}


#' Creates a dataset with at most count elements from this dataset
#'
#' @param dataset A dataset
#' @param count Integer representing the number of elements of this dataset that
#'   should be taken to form the new dataset. If `count` is -1, or if `count` is
#'   greater than the size of this dataset, the new dataset will contain all
#'   elements of this dataset.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_take <- function(dataset, count) {
  dataset$take(count = as_integer_tensor(count))
}


#' Maps `map_func`` across this dataset.
#'
#' @param dataset A dataset
#' @param map_func A function mapping a nested structure of tensors (having
#'   shapes and types defined by [output_shapes()] and [output_types()] to
#'   another nested structure of tensors.
#' @param num_threads (Optional) An integer, representing the number of threads
#'   to use for processing elements in parallel. If not specified, elements will
#'   be processed sequentially without buffering.
#' @param output_buffer_size (Optional) An integer, representing the maximum
#'   number of processed elements that will be buffered when processing in
#'   parallel.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_map <- function(dataset, map_func, num_threads = NULL, output_buffer_size = NULL) {
  dataset$map(
    map_func = map_func,
    num_threads = as_integer_tensor(num_threads, tf$int32),
    output_buffer_size = as_integer_tensor(num_threads)
  )
}


#' Enumerate the elements of this dataset.
#'
#' Adds a counter to an iterable. So for each element in the dataset, a tuple is produced with (counter, element)
#'
#' @param dataset A dataset
#' @param start A integer, representing the start value for enumeration.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_enumerate <- function(dataset, start = 0) {
  dataset$enumerate(start = as_integer_tensor(start))
}


#' Creates a dataset that skips count elements from this dataset
#'
#' @param dataset A dataset
#' @param count An integer, representing the number of elements of this dataset
#'   that should be skipped to form the new dataset. If count is greater than
#'   the size of this dataset, the new dataset will contain no elements. If
#'   count is -1, skips the entire dataset.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_skip <- function(dataset, count) {
  dataset$skip(count = as_integer_tensor(count))
}



with_session <- function(f, session = NULL) {
  if (is.null(session))
    session <- tf$get_default_session()
  if (is.null(session)) {
    session <- tf$Session()
    on.exit(session$close(), add = TRUE)
  }
  f(session)
}





