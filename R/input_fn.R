

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
#' @return An input_fn suitable for use with tfestimators [train][tfestimators::train.tf_estimator], [evaluate][tfestimators::evaluate.tf_estimator], and
#'   [predict][tfestimators::predict.tf_estimator] methods
#'
#' @export
input_fn_from_dataset <- function(dataset, features, response) {

  # validate/retreive column names
  col_names <- column_names(dataset)

  # get tidyselect_data for overscope
  tidyselect <- asNamespace("tidyselect")
  exports <- getNamespaceExports(tidyselect)
  tidyselect_data <- mget(exports, tidyselect, inherits = TRUE)

  # evaluate features (use tidyselect overscope)
  eq_features <- enquo(features)
  environment(eq_features) <- as_overscope(eq_features, data = tidyselect_data)
  feature_names <- vars_select(col_names, !! eq_features)

  # evaluate response (use tidyselect overscope)
  eq_response <- enquo(response)
  environment(eq_response) <- as_overscope(eq_response, data = tidyselect_data)
  response_name <- vars_select(col_names, !! eq_response)
  if (length(response_name) != 1)
    stop("More than one response column specified: ", paste(response_name))

  # return function which yields a features/response iterator for the dataset
  function(estimator) {
    function() {
      batch_from_dataset(
        dataset = dataset,
        features = feature_names,
        response = response_name,
        names = NULL,
        named_features = !inherits(estimator, "tf_custom_estimator")
      )
    }
  }
}

#' @export
input_fn.tensorflow.python.data.ops.dataset_ops.Dataset <- function(object, features, response, ...) {

  # alias dataset
  dataset <- object

  # validate/retreive column names
  col_names <- column_names(dataset)

  # get tidyselect_data for overscope
  tidyselect <- asNamespace("tidyselect")
  exports <- getNamespaceExports(tidyselect)
  tidyselect_data <- mget(exports, tidyselect, inherits = TRUE)

  # evaluate features (use tidyselect overscope)
  eq_features <- enquo(features)
  environment(eq_features) <- as_overscope(eq_features, data = tidyselect_data)
  feature_names <- vars_select(col_names, !! eq_features)

  # evaluate response (use tidyselect overscope)
  eq_response <- enquo(response)
  environment(eq_response) <- as_overscope(eq_response, data = tidyselect_data)
  response_name <- vars_select(col_names, !! eq_response)
  if (length(response_name) != 1)
    stop("More than one response column specified: ", paste(response_name))

  input_fn_from_dataset(dataset, feature_names, response_name)
}


column_names <- function(dataset) {
  if (!is.list(dataset$output_shapes) || is.null(names(dataset$output_shapes)))
    stop("Dataset does not have named outputs", call. = FALSE)
  names(dataset$output_shapes)
}



