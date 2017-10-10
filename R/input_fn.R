

#' Construct an input function from a dataset
#'
#' @param dataset A dataset
#' @param features The names of feature variables to be used.
#' @param response The name of the response variable.
#' @param ... Unused
#'
#' @details Creating an input_fn from a dataset requires that the dataset
#'   consist of a set of named output tensors (e.g. like the dataset
#'   produced by the [csv_dataset()] function).
#'
#' @return An input_fn suitable for use with tfestimators train, evaluate, and
#'   predict methods
#'
#'
#' @export
input_fn.tensorflow.python.data.ops.dataset_ops.Dataset <- function(dataset, features, response, ...) {

  # validate/retreive column names
  if (!is.list(dataset$output_shapes) || is.null(names(dataset$output_shapes)))
    stop("Creating an input_fn requires a dataset with named outputs")
  col_names <- names(dataset$output_shapes)

  # get tidyselect_data for overscope
  tidyselect <- asNamespace("tidyselect")
  exports <- getNamespaceExports(tidyselect)
  tidyselect_data <- mget(exports, tidyselect, inherits = TRUE)

  # evaluate features (use tidyselect overscope)
  eq_features <- enquo(features)
  environment(eq_features) <- as_overscope(eq_features, data = tidyselect_data)
  features <- vars_select(col_names, !! eq_features)
  feature_cols <- match(features, col_names)

  # evaluate response (use tidyselect overscope)
  if (!missing(response)) {
    eq_response <- enquo(response)
    environment(eq_response) <- as_overscope(eq_response, data = tidyselect_data)
    response <- vars_select(col_names, !! eq_response)
    if (length(response) != 1)
      stop("More than one response column specified: ", paste(response))
    response_col <- match(response, col_names)
  }

  # map dataset into input_fn compatible tensors
  input_fn_dataset <- dataset %>%
    dataset_map(function(record) {
      record_features <- record[feature_cols]
      names(record_features) <- features
      if (!is.null(response))
        record_response <- record[[response_col]]
      else
        record_response <- NULL
      tuple(record_features, record_response)
    })

  # return function which yields the iterator for the dataset
  function(estimator) {
    if (inherits(estimator, "tf_custom_estimator"))
      NULL
    else {
      function() {
        iter <- one_shot_iterator(input_fn_dataset)
        iter$get_next()
      }
    }
  }
}

