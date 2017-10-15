
#' Tensor(s) for drawing batches from a dataset
#'
#' Tensor or list(s) of tensors (e.g. for features and response)
#' that yield the next batch of data each time they are evaluated.
#'
#' @param dataset A dataset
#'
#' @param features Features to include.
#'
#' @param response Response variable.
#'
#' @param named `TRUE` to assign names to list elements when `features` are
#'  specified (defaults to `x` and `y`, provide a character vector to use
#'  alterate names).
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
#' using the `named` argument (pass `FALSE` to return a list with no names).
#'
#' When `features` is not specified the tensors will conform to the
#' shape and types of the dataset (see [output_shapes()] and [output_types()]).
#'
#' @section Batch Iteration:
#'
#' In many cases you won't need to explicitly iterate over the batches,
#' rather, you will pass the tensors to another function that will perform
#' the evaluation (e.g. the Keras `layer_input()` and `compile()` functions).
#'
#' If you do need to perform iteration manually by evaluating the tensors, there
#' are a couple of possible approaches to controlling/detecting when iteration should
#' end.
#'
#' One approach is to create a dataset that yields batches infinitely (traversing
#' the dataset multiple times with different batches randomly drawn). In this case you'd
#' use another mechanism like a global step counter or check for a learning plateau.
#'
#' Another approach is to iteration is to detect when all batches have been yielded
#' from the dataset. When the batch tensor reaches the end of iteration a runtime
#' error will occur. You can catch and ignore the error when it occurs by wrapping
#' your iteration code in the `with_dataset_iterator()` function.
#'
#' See the examples below for a demonstration of each of these methods of iteration.
#'
#' @examples \dontrun{
#'
#' # iteration with 'infinite' dataset and explicit step counter
#'
#' library(tfdatasets)
#' dataset <- csv_dataset("training.csv") %>%
#'   dataset_shuffle(5000) %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat()
#' batch <- batch_from_dataset(dataset, features = c(mpg, disp), response = cyl)
#' steps <- 200
#' for (i in 1:steps) {
#'   # use batch$x and batch$y tensors
#' }
#'
#' # iteration that detects and ignores end of iteration error
#'
#' library(tfdatasets)
#' dataset <- csv_dataset("training.csv") %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat(10)
#' batch <- batch_from_dataset(dataset, features = c(mpg, disp), response = cyl)
#' with_dataset_iterator({
#'   while(TRUE) {
#'     # use batch$x and batch$y tensors
#'   }
#' })
#' }
#'
#' @seealso [input_fn_from_dataset()] for use with \pkg{tfestimators}.
#'
#' @export
batch_from_dataset <- function(dataset, features = NULL, response = NULL,
                               named = TRUE, named_features = FALSE) {

  # validate dataset
  if (!inherits(dataset, "tensorflow.python.data.ops.dataset_ops.Dataset"))
    stop("Provided dataset is not a TensorFlow Dataset")

  # get tidyselect_data for overscope
  tidyselect <- asNamespace("tidyselect")
  exports <- getNamespaceExports(tidyselect)
  tidyselect_data <- mget(exports, tidyselect, inherits = TRUE)

  # get features if specified
  if (!missing(features)) {
    col_names <- column_names(dataset)
    eq_features <- enquo(features)
    environment(eq_features) <- as_overscope(eq_features, data = tidyselect_data)
    feature_col_names <- vars_select(col_names, !! eq_features)
    feature_cols <- match(feature_col_names, col_names)
  } else {
    col_names <- NULL
    feature_cols <- NULL
  }

  # get response if specified
  response_col <- NULL
  if (!missing(response)) {
    if (is.null(col_names))
      stop("You must specify features if you specify a response")
    eq_response <- enquo(response)
    environment(eq_response) <- as_overscope(eq_response, data = tidyselect_data)
    response_name <- vars_select(col_names, !! eq_response)
    if (length(response_name) > 0) {
      if (length(response_name) != 1)
        stop("Invalid response column: ", paste(response_name))
      response_col <- match(response_name, col_names)
    }
  }

  # return the right kind of iterator depending on the request

  # transform for feature/response selection if requested
  if (!is.null(feature_cols)) {
    dataset <- dataset %>%
      dataset_map(function(record) {

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

        # return with or without response
        if (!is.null(response_col))
          tuple(record_features, record[[response_col]])
        else
          record_features
      })

    # get batch tensor
    iterator <- dataset$make_one_shot_iterator()
    batch <- iterator$get_next()

    # if there is no response then reshape to include a NULL response
    if (is.null(response_col))
      batch <- list(batch, NULL)

    # resolve names
    if (isTRUE(named))
      names <- c("x", "y")
    else if (is.character(named)) {
      if (length(named) != 2)
        stop("names must be TRUE, FALSE, or a 2-element character vector")
      names <- named
    } else {
      names <- NULL
    }

    # apply names and return batch
    names(batch) <- names
    batch

  } else {

    iterator <- dataset$make_one_shot_iterator()
    iterator$get_next()

  }
}


#' Execute code that checks for end of dataset iteration
#'
#' @param expr Expression to execute
#'
#' @details  When a dataset iterator reaches the end, an out of range runtime error
#'   will occur. You can catch and ignore the error when it occurs by wrapping
#'   your iteration code in the `with_dataset_iterator()` (see the example
#'   below for an illustration).
#'
#' @examples \dontrun{
#' library(tfdatasets)
#' dataset <- csv_dataset("training.csv") %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat(10)
#'
#' batch <- batch_from_dataset(dataset, features = c(mpg, disp), response = cyl)
#' with_dataset_iterator({
#'   while(TRUE) {
#'     # use batch$x and batch$y tensors
#'   }
#' })
#' }
#'
#' @export
with_dataset_iterator <- function(expr) {
  tryCatch({
    force(expr)
  },
  error = function(e) {
    last_error <- py_last_error()
    if (is.null(last_error) || !identical(last_error$type, "OutOfRangeError"))
      stop(e)
  })
}









