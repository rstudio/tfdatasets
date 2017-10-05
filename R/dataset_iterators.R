


#' Creates an iterator for enumerating the elements of this dataset.
#'
#' The returned iterator will be initialized automatically. A "one-shot"
#' iterator does not currently support re-initialization.
#'
#' @param dataset A dataset
#'
#' @return An iterator over the elements of this dataset
#'
#' @family dataset iterators
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
#' @param dataset A dataset
#'
#' @param shared_name (Optional) If non-empty, this iterator will be shared under the
#' given name across multiple sessions that share the same devices (e.g. when
#' using a remote server).
#'
#' @return An iterator over the elements of this dataset
#'
#' @family dataset iterators
#'
#' @export
initializable_iterator <- function(dataset, shared_name = NULL) {
  dataset$make_initializable_iterator(shared_name)
}


#' Get the next values from an iterator
#'
#' Returns the evaluation of the next tensor(s) available from the iterator or
#' the sentinel `completed` value if there are no more tensors to iterate.
#'
#' @param iter Dataset iterator
#' @param session TensorFlow session to evaluate iterator within
#' @param completed Sentinel value to return from `iterator_next()` if the
#'   iteration completes (defaults to `NULL` but can be any value you specify).
#'
#' @return Nested structure of tensors containing the next element.
#'
#' @family dataset iterators
#'
#' @export
iterator_next <- function(iter, session, completed = NULL) {
  tryCatch({
    session$run(iter$get_next())
  },
  error = function(e) {
    last_error <- py_last_error()
    if (!is.null(last_error) && identical(last_error$type, "OutOfRangeError"))
      completed
    else
      stop(e)
  })
}





