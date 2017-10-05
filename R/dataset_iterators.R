


#' Creates an iterator for enumerating the elements of this dataset.
#'
#' The returned iterator will be initialized automatically. A "one-shot"
#' iterator does not currently support re-initialization.
#'
#' @return An iterator over the elements of this dataset
#'
#' @export
one_shot_iterator <- function(dataset) {
  dataset$make_one_shot_iterator()
}


#' Creates an Iterator for enumerating the elements of this dataset.
#'
#' The returned iterator will be in an uninitialized state, and you must run the
#' iterator.initializer operation before using it.
#'
#' @param shared_name (Optional) If non-empty, this iterator will be shared under the
#' given name across multiple sessions that share the same devices (e.g. when
#' using a remote server).
#'
#' @return An iterator over the elements of this dataset
#'
#' @export
initializable_iterator <- function(dataset, shared_name = NULL) {
  dataset$make_initializable_iterator(shared_name)
}




