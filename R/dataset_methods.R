

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


#' Map a function across a dataset.
#'
#' @param dataset A dataset
#' @param map_func A function mapping a nested structure of tensors (having
#'   shapes and types defined by [output_shapes()] and [output_types()] to
#'   another nested structure of tensors.
#' @param num_parallel_calls (Optional) An integer, representing the
#'   number of elements to process in parallel If not specified, elements will
#'   be processed sequentially.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_map <- function(dataset, map_func, num_parallel_calls = NULL) {
  dataset$map(
    map_func = map_func,
    num_parallel_calls = as_integer_tensor(num_parallel_calls, tf$int32)
  )
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



#' Prepare a dataset for analysis
#'
#' Transform a dataset with named columns into a list with features (`x`) and
#' response (`y`) elements.
#'
#' @inheritParams dataset_map
#'
#' @param dataset A dataset
#'
#' @param x Features to include. When `named_features` is `FALSE` all features
#'   will be stacked into a single tensor so must have an identical data type.
#'
#' @param y (Optional). Response variable.
#'
#' @param named `TRUE` to name the dataset elements "x" and "y", `FALSE` to
#'   not name the dataset elements.
#'
#' @param named_features `TRUE` to yield features as a named list; `FALSE` to
#'   stack features into a single array. Note that in the case of `FALSE` (the
#'   default) all features will be stacked into a single 2D tensor so need to
#'   have the same underlying data type.
#'
#' @return A dataset. The dataset will have a structure of either:
#'
#'   - When `named_features` is `TRUE`: `list(x = list(feature_name = feature_values, ...), y = response_values)`
#'
#'   - When `named_features` is `FALSE`: `list(x = features_array, y = response_values)`,
#'     where `features_array` is a Rank 2 array of `(batch_size, num_features)`.
#'
#' Note that the `y` element will be omitted when `y` is `NULL`.
#'
#' @seealso [input_fn()][input_fn.tensorflow.python.data.ops.dataset_ops.Dataset()] for use with \pkg{tfestimators}.
#'
#' @export
dataset_prepare <- function(dataset, x, y = NULL, named = TRUE, named_features = FALSE,
                            num_parallel_calls = NULL) {

  # validate dataset
  if (!inherits(dataset, "tensorflow.python.data.ops.dataset_ops.Dataset"))
    stop("Provided dataset is not a TensorFlow Dataset")

  # get tidyselect_data for overscope
  tidyselect <- asNamespace("tidyselect")
  exports <- getNamespaceExports(tidyselect)
  tidyselect_data <- mget(exports, tidyselect, inherits = TRUE)

  # default to null response_col
  response_col <- NULL

  # get features
  col_names <- column_names(dataset)
  eq_features <- enquo(x)
  environment(eq_features) <- as_overscope(eq_features, data = tidyselect_data)

  # attempt use of tidyselect. if there is an error it could be because 'x'
  # is a formula. in that case attempt to parse the formula
  feature_col_names <- tryCatch({
    vars_select(col_names, !! eq_features)
  },
  error = function(e) {
    if (is_formula(x)) {
      parsed <- parse_formula(x)
      if (!is.null(parsed$response))
        response_col <<- match(parsed$response, col_names)
      parsed$features
    } else {
      stop(e$message, call. = FALSE)
    }
  })

  # get column indexes
  feature_cols <- match(feature_col_names, col_names)

  # get response if specified
  if (!missing(y) && is.null(response_col)) {
    eq_response <- enquo(y)
    environment(eq_response) <- as_overscope(eq_response, data = tidyselect_data)
    response_name <- vars_select(col_names, !! eq_response)
    if (length(response_name) > 0) {
      if (length(response_name) != 1)
        stop("Invalid response column: ", paste(response_name))
      response_col <- match(response_name, col_names)
    }
  }

  # transform for feature/response selection

  dataset <- dataset %>%

    dataset_map(num_parallel_calls = num_parallel_calls, function(record) {

      # select features
      record_features <- record[feature_cols]

      # apply names to features if named
      if (named_features) {

        names(record_features) <- feature_col_names

        # otherwise stack features into a single tensor
      } else {
        record_features <- unname(record_features)
        # determine the axis based on the shape of the tensor
        # (unbatched tensors will be scalar with no shape,
        #  so will stack on axis 0)
        shape <- record_features[[1]]$get_shape()$as_list()
        axis <- length(shape)
        record_features <- tf$stack(record_features, axis = axis)
      }

      # massage the record into the approriate structure
      if (!is.null(response_col)) {
        record <- list(record_features, record[[response_col]])
        if (named)
          names(record) <- c("x", "y")
      }
      else {
        record <- list(record_features)
        if (named)
          names(record) <- c("x")
      }

      # return the record
      record
    })
}



