

#' Repeats a dataset count times.
#'
#' @param count (Optional.) An integer value representing the number of times
#'   the elements of this dataset should be repeated. The default behavior (if
#'   `count` is `NULL` or `-1`) is for the elements to be repeated indefinitely.
#'
#' @return A dataset
#'
#' @export
dataset_repeat <- function(dataset, count = NULL) {
  dataset$`repeat`(as_tensor_int64(count))
}
