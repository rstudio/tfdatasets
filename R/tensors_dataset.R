
resolve_tensors <- function(tensors) {
  # convert unnamed list into tuple
  if (is.list(tensors) && is.null(names(tensors)))
    tensors <- tuple(tensors)

  # covert data frame to dict
  if (is.data.frame(tensors)) {
    tensors <- as.list(tensors)
  }

  tensors
}

#' Creates a dataset with a single element, comprising the given tensors.
#'
#' @param tensors A nested structure of tensors.
#'
#' @return A dataset.
#'
#' @family tensor datasets
#'
#' @export
tensors_dataset <- function(tensors) {
  validate_tf_version()

  as_tf_dataset(
    tf$data$Dataset$from_tensors(tensors = resolve_tensors(tensors))
  )
}

#' Creates a dataset whose elements are slices of the given tensors.
#'
#' @param tensors A nested structure of tensors, each having the same size in
#'   the 0th dimension.
#'
#' @return A dataset.
#'
#' @family tensor datasets
#'
#' @export
tensor_slices_dataset <-function(tensors) {

  validate_tf_version()

  as_tf_dataset(
    tf$data$Dataset$from_tensor_slices(tensors = resolve_tensors(tensors))
  )
}

#' Splits each rank-N `tf$SparseTensor` in this dataset row-wise.
#'
#' @param sparse_tensor A `tf$SparseTensor`.
#'
#' @return A dataset of rank-(N-1) sparse tensors.
#'
#' @family tensor datasets
#'
#' @export
sparse_tensor_slices_dataset <- function(sparse_tensor) {
  validate_tf_version()
  as_tf_dataset(
    tf$data$Dataset$from_sparse_tensor_slices(sparse_tensor = sparse_tensor)
  )
}




