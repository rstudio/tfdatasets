


#' Creates a dataset of a step-separated range of values.
#'
#' @param from Range start
#' @param to Range end (exclusive)
#' @param by Increment of the sequence
#'
#' @export
range_dataset <- function(from = 0, to = 0, by = 1) {

  # cast to correct integer types
  from <- as_integer_tensor(from)
  to <- as_integer_tensor(to)
  by <- as_integer_tensor(by)

  # create dataset
  as_tf_dataset(
    tf$data$Dataset$range(from, to, by)
  )
}
