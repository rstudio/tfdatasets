

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
  object$output_types
}


#' @rdname output_types
#' @export
output_shapes <- function(object) {
  object$output_shapes
}



