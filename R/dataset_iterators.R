

#' Create an iterator for enumerating the elements of this dataset.
#'
#' The returned iterator will be initialized automatically. The iterator is
#' "one-shot" (i.e. it performs a single pass over the dataset and cannot be
#' re-initialized)
#'
#' @param dataset A dataset
#' @param features Featured to include. When `features` is specified the
#'   iterator value will be either:
#'   - `list(list(feature_name = feature_value, ...), response_value)` when
#'   `named_features` is `TRUE`; or
#'   - `list(features_array, response_value)` when `named_features` is `FALSE`,
#'     where `features_array` is a Rank 2 array of (batch_size, num_features).
#' @param response Response variable (required from `features` is specified).
#'
#' @details Pass the iterator to [next_element()] in order to retreive the value
#'   of the next element, or [next_element_tensor()] to retreive a tensor that
#'   can be evaluated repeatedly to retreive the value of the next element.
#'
#' @return An iterator over the elements of this dataset.
#'
#' @export
iterator_from_dataset <- function(dataset, features = NULL, response = NULL, named_features = TRUE) {

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
          record_features <- tf$stack(record_features, axis = 1L)
          tuple(record_features, record_response)
        }
      })
  }

  # create and return iterator
  dataset$make_one_shot_iterator()
}



#' Get the next element from an iterator
#'
#' @param iterator Dataset iterator
#' @param session TensorFlow session to evaluate iterator within. If you
#'   pass `NULL` then the default session will be used if it's been
#'   previously set.
#' @param completed Sentinel value to return from `next_element()` if the
#'   iteration completes (defaults to `NULL` but can be any value you specify).
#'
#' @return For `next_element()`, the value of the next element, or `completed`
#'   if there are no more elements available. For `next_element_tensor()`, a
#'   tensor which can be evaluated repeatedly to obtain the next element.
#'
#' @section Value Iteration:
#' To iterate using `next_element()`, check for a `NULL` return value
#' (or other custom value specified via `completed`) to detect the
#' end of the iteration. For example:
#'
#' ```r
#' library(tfdatasets)
#' library(tensorflow)
#' sess <- tf$Session()
#' dataset <- tensors_dataset(1:10)
#' iterator <- iterator_from_dataset(dataset)
#' while(!is.null(value <- next_element(iterator, sess))) {
#'   # do something with the value
#' }
#' ```
#'
#' Note that a TensorFlow session is required to evaluate element values. If you
#' pass `NULL` then the default TensorFlow session will be used if it's been
#' previously set.
#'
#' @section Tensor Iteration:
#' If you use are iterating based on evaluating the tensor returned from
#' `next_element_tensor()` a runtime error will occur when the iterator
#' has exhausted all available elements. You can use the `is_out_of_range_error()`
#' to distinguish this error from other errors which may have occurred. For example:
#'
#' ```r
#' library(tfdatasets)
#' dataset <- tensors_dataset(1:10)
#' iterator <- iterator_from_dataset(dataset)
#' next_element <- next_element_tensor(iterator)
#' tryCatch({
#'   while(TRUE) {
#'     # do something with the next_element tensor
#'   }
#' },
#' error = function(e) {
#'   if (!is_out_of_range_error())
#'     stop(e)
#' })
#' ```
#'
#' @family dataset iterators
#'
#' @export
next_element <- function(iterator, session = NULL, completed = NULL) {

  # if the session is NULL then look for a default session
  if (is.null(session)) {
    session <- tf$get_default_session()
    if (is.null(session))
      stop("No session specified and no default session available.")
  }

  tryCatch({
    session$run(next_element_tensor(iterator))
  },
  error = function(e) {
    if (is_out_of_range_error())
      completed
    else
      stop(e)
  })
}


#' @rdname next_element
#' @export
next_element_tensor <- function(iterator) {
  iterator$get_next()
}


#' @rdname next_element
#' @export
is_out_of_range_error <- function() {
  last_error <- py_last_error()
  !is.null(last_error) && identical(last_error$type, "OutOfRangeError")
}





