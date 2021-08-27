


#' Creates a dataset of a step-separated range of values.
#'
#' @param from Range start
#' @param to Range end (exclusive)
#' @param by Increment of the sequence
#'
#' @param ... ignored
#'
#' @param dtype Output dtype. (Optional, default: `tf$int64`).
#'
#' @export
range_dataset <- function(from = 0, to = 0, by = 1, ..., dtype = tf$int64) {


  # cast to correct integer types
  from <- as_integer_tensor(from)
  to <- as_integer_tensor(to)
  by <- as_integer_tensor(by)

  args <- list(from, to, by)
  # args <- args[vapply(args, is.null, TRUE)]

  if(!missing(dtype))
    args$output_type <- dtype

  # create dataset
  as_tf_dataset(
    do.call(tf$data$Dataset$range, args)
  )
}

  # TODO: make this more like `seq.default()` and python's `range()`:
  # if only 1 arg provided, take it as the `to` field:

  #  @param *args follows the same semantics as python's xrange.
  #  len(args) == 1 -> start = 0, stop = args[0], step = 1.
  #  len(args) == 2 -> start = args[0], stop = args[1], step = 1.
  #  len(args) == 3 -> start = args[0], stop = args[1], step = args[2].
