

#' Dataset output types and shapes
#'
#' @param dataset A dataset
#'
#' @return `output_types()` returns the type of each component of an element of
#'   this dataset; `output_shapes()` returns the shape of each component of an
#'   element of this dataset.
#'
#' @export
output_types <- function(dataset) {
  dataset$output_types
}


#' @rdname output_types
#' @export
output_shapes <- function(dataset) {
  dataset$output_shapes
}



