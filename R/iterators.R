



#' Creates an iterator for enumerating the elements of this dataset.
#'
#' @section Initialization:
#'   For `make_iterator_one_shot()`, the returned
#'   iterator will be initialized automatically. A "one-shot" iterator does not
#'   currently support re-initialization.
#'
#'   For `make_iterator_initializable()`,
#'   the returned iterator will be in an uninitialized state, and you must run
#'   the object returned from [iterator_initializer()] before using it.
#'
#'   For `make_iterator_from_structure()`, the returned iterator is not bound
#'   to a particular dataset, and it has no initializer. To initialize the
#'   iterator, run the operation returned by [iterator_make_initializer()].
#'
#' @param dataset A dataset
#' @param output_types A nested structure of tf$DType objects corresponding to
#'   each component of an element of this iterator.
#' @param output_shapes (Optional) A nested structure of tf$TensorShape objects
#'   corresponding to each component of an element of this dataset. If omitted,
#'   each component will have an unconstrainted shape.
#' @param string_handle A scalar tensor of type string that evaluates
#'  to a handle produced by the [iterator_string_handle()] method.
#' @param shared_name (Optional) If non-empty, the returned iterator will be
#'   shared under the given name across multiple sessions that share the same
#'   devices (e.g. when using a remote server).
#'
#' @return An Iterator over the elements of this dataset.
#'
#' @family iterator functions
#'
#' @name make-iterator
#' @export
make_iterator_one_shot <- function(dataset) {
  dataset$make_one_shot_iterator()
}


#' @rdname make-iterator
#' @export
make_iterator_initializable <- function(dataset, shared_name = NULL) {
  dataset$make_initializable_iterator(shared_name = shared_name)
}



#' @rdname make-iterator
#' @export
make_iterator_from_structure <- function(output_types, output_shapes = NULL,
                                         shared_name = NULL) {
  tf$data$Iterator$from_structure(
    output_types = output_types,
    output_shapes = output_shapes,
    shared_name = shared_name
  )
}

#' @rdname make-iterator
#' @export
make_iterator_from_string_handle <- function(string_handle, output_types,
                                             output_shapes = NULL) {
  tf$data$Iterator$from_string_handle(
    string_handle = string_handle,
    output_types = output_types,
    output_shapes = output_shapes
  )
}

#' Get next element from iterator
#'
#' Returns a nested list of tensors that when evaluated will yield
#' the next element(s) in the dataset.
#'
#' @param iterator An iterator
#' @param name (Optional) A name for the created operation.
#'
#' @return A nested list of tensors
#'
#' @family iterator functions
#'
#' @export
iterator_get_next <- function(iterator, name = NULL) {
  iterator$get_next()
}


#' An operation that should be run to initialize this iterator.
#'
#' @param iterator An iterator
#'
#' @family iterator functions
#'
#' @export
iterator_initializer <- function(iterator) {
  iterator$initializer
}

#' String-valued tensor that represents this iterator
#'
#' @inheritParams iterator_get_next
#'
#' @return Scalar tensor of type string
#'
#' @family iterator functions
#'
#' @export
iterator_string_handle <- function(iterator, name = NULL) {
  iterator$string_handle(name = name)
}



#' Create an operation that can be run to initialize this iterator
#'
#' @param iterator An iterator
#' @param dataset A dataset
#' @param name (Optional) A name for the created operation.
#'
#' @return A tf$Operation that can be run to initialize this iterator on the
#'   given dataset.
#'
#' @family iterator functions
#'
#' @export
iterator_make_initializer <- function(iterator, dataset, name = NULL) {
  iterator$make_initializer(dataset = dataset, name = name)
}








