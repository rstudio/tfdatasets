

#' Construct a Keras generator function from a dataset
#'
#' @param dataset A dataset
#' @param features The names of feature variables to be used.
#' @param response The name of the response variable.
#' @param filter Optional filter function which
#' @param session TensorFlow session to evaluate values within. If
#'   "default", then the current Keras TensorFlow session will be
#'   used (if available) and failing that the default TensorFlow
#'   session as given by `tf$get_default_session()`.
#'
#' @details Creating a generator from a dataset requires that the dataset
#'   consist of a set of named output tensors (e.g. like the dataset produced by
#'   the [csv_dataset()] function).
#'
#' @return A generator function suitable for use with the keras
#'   [fit_generator][keras::fit_generator],
#'   [evaluate_generator](keras::evaluate_generator), and
#'   [predict_generator](keras::predict_generator) functions.
#'
#' @export
generator_from_dataset <- function(dataset, features, response, session = "default") {

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

  # create iterator
  iterator <- iterator_from_dataset(
    dataset = dataset,
    features = feature_names,
    response = response_name,
    named_features = FALSE
  )

  # get reference to keras tensorflow backend
  if (reticulate::py_module_available("keras")) {
    keras <- reticulate::import("keras")
    tf_backend <- keras$backend$tensorflow_backend
  } else {
    tf_backend <- NULL
  }

  # return function that yields elements progressively
  function() {
    # resolve session
    if (identical(session, "default") && !is.null(tf_backend))
      session <- tf_backend$`_SESSION`
    if (is.null(session))
      session <- tf$get_default_session()

    # yield next element
    next_element(iterator, session = session)
  }
}




