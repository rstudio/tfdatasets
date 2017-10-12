
#' Tensors for iterating over batches of a dataset.
#'
#' Tensor or list(s) of tensors (e.g. for named features and response)
#' that yield the next batch of data each time they are evaluated.
#'
#' @param dataset A dataset
#'
#' @param features Features to include.
#'
#' @param response Response variable (required when `features` is specified).
#'
#' @param names `TRUE` to yield features as a named list; `FALSE`
#'   to stack features into a single array. Note that in the case of `FALSE`
#'   all features will be stacked into a single 2D tensor so need to have the
#'   same underlying data type.
#'
#' @return Tensors that can be evaluated to yield the next batch of training data.
#'
#' When `features` is specified the tensor will have a structure of either:
#'   - `list(list(feature_name = feature_value, ...), response_value)` when
#'   `names` is `TRUE`; or
#'   - `list(features_array, response_value)` when `names` is `FALSE`,
#'     where `features_array` is a Rank 2 array of (batch_size, num_features).
#'
#' When `features` is not specified the tensors will conform to the
#' shape and types of the dataset (see [output_shapes()] and [output_types()]).
#'
#' @section Batch Iteration:
#'
#' In many cases you won't need to directly evaluate the batch tensors,
#' rather, you will pass the tensors to another function that will perform
#' the evaluation (e.g. the Keras `layer_input()` and `compile()` functions).
#'
#' If you do need to perform iteration manually by evaluating the tensors,
#' a runtime error will occur when the iterator has exhausted all available elements.
#' You can use the `is_out_of_range_error()` to distinguish this error from other
#' errors which may have occurred. For example:
#'
#' ```r
#' library(tfdatasets)
#' sess <- tf$Session()
#' dataset <- tensors_dataset(1:10)
#' batch <- batch_from_dataset(dataset)
#' tryCatch({
#'   while(TRUE) {
#'     batch_values <- sess$run(batch)
#'     print(batch_values)
#'   }
#' },
#' error = function(e) {
#'   if (!is_out_of_range_error())
#'     stop(e)
#' })
#' ```
#'
#' @export
batch_from_dataset <- function(dataset, features = NULL, response = NULL, names = FALSE) {

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
        if (names) {
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
  }

  # create iterator and return get_next() tensor
  iterator <- dataset$make_one_shot_iterator()
  iterator$get_next()
}

#' @rdname batch_from_dataset
#' @export
is_out_of_range_error <- function() {
  last_error <- py_last_error()
  !is.null(last_error) && identical(last_error$type, "OutOfRangeError")
}





