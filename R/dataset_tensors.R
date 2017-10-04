


#' Creates a dataset with a single element, comprising the given tensors.
#'
#' @param tensors A nested structure of tensors.
#'
#' @return A dataset.
#'
#' @family tensor datasets
#'
#' @export
dataset_from_tensors <- function(tensors) {
  tf$contrib$data$Dataset$from_tensors(
    tensors = tensors
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
dataset_from_tensor_slices <-function(tensors) {
  tf$contrib$data$Dataset$from_tensor_slices(
    tensors = tensors
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
dataset_from_sparse_tensor_slices <- function(sparse_tensor) {
  tf$contrib$data$Dataset$from_sparse_tensor_slices(
    sparse_tensor = sparse_tensor
  )
}




