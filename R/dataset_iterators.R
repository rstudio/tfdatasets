
#' Tensor for drawing batches from a dataset
#'
#' Tensor or list(s) of tensors (e.g. for features and response)
#' that yield the next batch of data each time they are evaluated.
#'
#' @param dataset A dataset
#'
#' @param features Features to include.
#'
#' @param response Response variable (required when `features` is specified).
#'
#' @param names Names to assign list elements when `features` and `response` are
#'  specified (defaults to `x` and `y` for features and response respectively) .
#'
#' @param named_features `TRUE` to yield features as a named list; `FALSE`
#'   to stack features into a single array. Note that in the case of `FALSE`
#'   (the default) all features will be stacked into a single 2D tensor
#'   so need to have the same underlying data type.
#'
#' @return Tensor(s) that can be evaluated to yield the next batch of training data.
#'
#' When `features` is specified the tensors will have a structure of either:
#'   - `list(x = list(feature_name = feature_values, ...), y = response_values)` when
#'   `named_features` is `TRUE`; or
#'   - `list(x = features_array, y = response_values)` when `named_features` is `FALSE`,
#'     where `features_array` is a Rank 2 array of `(batch_size, num_features)`.
#'
#' Note that by default list elements are named `x` and `y`. This can be customized
#' using the `names` argument (pass `NULL` to return a list with no names).
#'
#' When `features` is not specified the tensors will conform to the
#' shape and types of the dataset (see [output_shapes()] and [output_types()]).
#'
#' @section Batch Iteration:
#'
#' In many cases you won't need to explicitly iterate over the batches,
#' rather, you will pass the batch tensors to another function that
#' evaluates them as part of a training loop (e.g. the Keras
#' `layer_input()` and `compile()` functions).
#'
#' If you do need to iterate explicitly over the batches, you can use
#' the [for_each_batch()] function. For example:
#'
#' ```r
#' library(tfdatasets)
#' dataset <- csv_dataset("training.csv") %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat(10)
#' batches <- batches_from_dataset(dataset)
#' for_each_batch(batches, function(batch) {
#'   # use batch$x and batch$y tensors
#'   # (return FALSE to stop iteration early)
#' })
#' ```
#'
#' @seealso [for_each_batch()]; [input_fn_from_dataset()] for use with \pkg{tfestimators}.
#'
#' @export
batches_from_dataset <- function(dataset, features = NULL, response = NULL,
                                 names = c("x", "y"), named_features = FALSE) {

  # get tidyselect_data for overscope
  tidyselect <- asNamespace("tidyselect")
  exports <- getNamespaceExports(tidyselect)
  tidyselect_data <- mget(exports, tidyselect, inherits = TRUE)

  # get features if specified
  if (!missing(features)) {
    if (missing(response))
      stop("You must specify a response if you specify features")
    col_names <- column_names(dataset)
    eq_features <- enquo(features)
    environment(eq_features) <- as_overscope(eq_features, data = tidyselect_data)
    feature_names <- vars_select(col_names, !! eq_features)
    feature_cols <- match(feature_names, col_names)
  } else {
    col_names <- NULL
    feature_cols <- NULL
  }

  # get response if specified
  if (!missing(response)) {
    if (is.null(col_names))
      stop("You must specify features if you specify a response")
    eq_response <- enquo(response)
    environment(eq_response) <- as_overscope(eq_response, data = tidyselect_data)
    response_name <- vars_select(col_names, !! eq_response)
    if (length(response_name) != 1)
      stop("More than one response column specified: ", paste(response_name))
    response_col <- match(response_name, col_names)
  } else {
    response_col <- NULL
  }


  # return the right kind of iterator depending on the request

  # transform for feature/response selection if requested
  if (!is.null(feature_cols)) {
    dataset <- dataset %>%
      dataset_map(function(record) {
        record_features <- record[feature_cols]
        record_response <- record[[response_col]]
        if (named_features) {
          names(record_features) <- feature_names
          tuple(record_features, record_response)
        } else {
          record_features <- unname(record_features)
          # determine the axis based on the shape of the tensor
          # (unbatched tensors will be scalar with no shape,
          #  so will stack on axis 0)
          shape <- record_features[[1]]$get_shape()$as_list()
          axis <- length(shape)
          record_features <- tf$stack(record_features, axis = axis)
          tuple(record_features, record_response)
        }
      })

    iterator <- dataset$make_one_shot_iterator()
    batch <- iterator$get_next()
    names(batch) <- names
    batch

  } else {

    iterator <- dataset$make_one_shot_iterator()
    iterator$get_next()

  }
}


#' Iterate over batch tensors
#'
#' @param batches Tensor(s) from [batch_tensor_from_dataset()]
#' @param f Function with signature `function(batch)` to
#'  call for each training batch .
#'
#' @note The provided function will be called until all
#'  available training batches have been yielded. You
#'  can return `FALSE` from the function to stop iteration
#'  early.
#'
#' @examples \dontrun{
#' library(tfdatasets)
#' dataset <- csv_dataset("training.csv") %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat(10)
#' batches <- batches_from_dataset(dataset)
#' for_each_batch(batches, function(batch) {
#'   # use batch$x and batch$y tensors
#'   # (return FALSE to stop iteration early)
#' }
#' }
#' @export
for_each_batch <- function(batches, f) {
  tryCatch({
    while(TRUE) {
      result <- f(batches)
      if (identical(result, FALSE))
        break
    }
  },
  error = function(e) {
    if (!out_of_range_error())
      stop(e)
  })
}


#' Check if the last TensorFlow error was OutOfRangeError
#'
#' Used to detect end of iteration when evaluating tensors returned by
#' [batches_from_dataset()].
#'
#' @return `TRUE` if the last error was OutOfRangeError
#'
#' @keywords internal
#'
#' @export
out_of_range_error <- function() {
  last_error <- py_last_error()
  !is.null(last_error) && identical(last_error$type, "OutOfRangeError")
}





