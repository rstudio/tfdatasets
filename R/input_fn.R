

#' Construct an input function from a dataset
#'
#' @param dataset A dataset
#' @param features The names of feature variables to be used.
#' @param response The name of the response variable.
#'
#' @details Creating an input_fn from a dataset requires that the dataset
#'   consist of a set of named output tensors (e.g. like the dataset
#'   produced by the [csv_dataset()] function).
#'
#' @return An input_fn suitable for use with tfestimators train, evaluate, and
#'   predict methods
#'
#' @export
input_fn.tensorflow.contrib.data.python.ops.dataset_ops.Dataset <- function(dataset, features, response) {

  # validate/retreive column names
  if (!is.list(dataset$output_shapes) || is.null(names(dataset$output_shapes)))
    stop("Creating an input_fn requires a dataset with named outputs")
  col_names <- names(dataset$output_shapes)

  # get the indexes of the features and response
  feature_cols <- match(features, col_names)
  if (any(is.na(feature_cols)))
    stop("Invalid feature columns specified: ", paste(features[is.na(feature_cols)], collapse = ", "))
  response_col <- match(response, col_names)
  if (length(response) != 1)
    stop("More than one response column specified: ", paste(response))
  if (is.na(response_col))
    stop("Invalid response column specified: ", response)

  # map dataset into input_fn compatible tensors
  input_fn_dataset <- dataset %>%
    dataset_map(function(record) {
      record_features <- record[feature_cols]
      names(record_features) <- features
      record_response <- record[[response_col]]
      tuple(record_features, record_response)
    })

  # return function which yields the iterator for the dataset
  # TODO: function(estimator) which supports custom estimators
  function() {
    iter <- one_shot_iterator(input_fn_dataset)
    iter$get_next()
  }
}

