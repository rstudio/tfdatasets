

#' Output types and shapes
#'
#' @param object A dataset or iterator
#'
#' @return `output_types()` returns the type of each component of an element of
#'   this object; `output_shapes()` returns the shape of each component of an
#'   element of this object
#'
#' @export
output_types <- function(object) {
  if (tensorflow::tf_version() < "2.0") {
    object$output_types
  } else {
    b <- next_batch(object)
    if (is.list(b)) {
      lapply(b, function(x) x$dtype)
    } else {
      b$dtype
    }
  }
}


#' @rdname output_types
#' @export
output_shapes <- function(object) {
  if (tensorflow::tf_version() < "2.0") {
    object$output_shapes
  } else {
    b <- next_batch(object)
    if (is.list(b)) {
      lapply(b, function(x) x$shape)
    } else {
      b$shape
    }
  }
}
