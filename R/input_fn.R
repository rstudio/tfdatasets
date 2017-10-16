

#' Construct a tfestimators input function from a dataset
#'
#' @param dataset A dataset
#' @param features The names of feature variables to be used.
#' @param response The name of the response variable.
#'
#' @details Creating an input_fn from a dataset requires that the dataset
#'   consist of a set of named output tensors (e.g. like the dataset
#'   produced by the [csv_dataset()] function).
#'
#' @return An input_fn suitable for use with tfestimators [train][tfestimators::train.tf_estimator],
#'   [evaluate][tfestimators::evaluate.tf_estimator], and [predict][tfestimators::predict.tf_estimator] methods
#'
#' @family reading datasets
#'
#' @rdname input_fn
#' @aliases input_fn
#' @export
input_fn.tensorflow.python.data.ops.dataset_ops.Dataset <- function(dataset, features, response = NULL) {

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
  response_name <- NULL
  if (!missing(response)) {
    eq_response <- enquo(response)
    environment(eq_response) <- as_overscope(eq_response, data = tidyselect_data)
    response_name <- vars_select(col_names, !! eq_response)
    if (length(response_name) > 0) {
      if (length(response_name) != 1)
        stop("More than one response column specified: ", paste(response_name))
    } else {
      response_name <- NULL
    }
  }

  # return function which yields a features/response iterator for the dataset
  function(estimator) {
    function() {

      # create new dataset
      dataset <- dataset %>%
        dataset_prepare(
          x = feature_names,
          y = response_name,
          named = FALSE,
          named_features = !inherits(estimator, "tf_custom_estimator")
        )

      # get the iterator
      batch <- next_batch(dataset)

      # add `NULL` response if needed
      if (is.null(response_name))
        batch[2] <- list(NULL)

      # return iterator
      batch
    }
  }
}



