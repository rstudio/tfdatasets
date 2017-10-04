


#' Creates a dataset with a single element, comprising the given tensors.
#'
#' @param tensors A nested structure of tensors.
#'
#' @template roxlate-create-dataset
#'
#' @return A dataset.
#'
#' @family tensor datasets
#'
#' @export
tensors_dataset <- function(tensors, shuffle = NULL, batch_size = NULL, repeated = NULL) {

  dataset <- tf$contrib$data$Dataset$from_tensors(tensors = tensors)

  if (!missing(shuffle))
    dataset <- dataset_shuffle(dataset, buffer_size = shuffle)
  if (!missing(repeated))
    dataset <- dataset_repeat(dataset, count = repeated)
  if (!missing(batch_size))
    dataset <- dataset_batch(dataset, batch_size)

  dataset
}

#' Creates a dataset whose elements are slices of the given tensors.
#'
#' @param tensors A nested structure of tensors, each having the same size in
#'   the 0th dimension.
#'
#' @template roxlate-create-dataset
#'
#' @return A dataset.
#'
#' @family tensor datasets
#'
#' @export
tensor_slices_dataset <-function(tensors, shuffle = NULL, batch_size = NULL, repeated = NULL) {

  dataset <- tf$contrib$data$Dataset$from_tensor_slices(tensors = tensors)

  if (!missing(shuffle))
    dataset <- dataset_shuffle(dataset, buffer_size = shuffle)
  if (!missing(repeated))
    dataset <- dataset_repeat(dataset, count = repeated)
  if (!missing(batch_size))
    dataset <- dataset_batch(dataset, batch_size)

  dataset
}

#' Splits each rank-N `tf$SparseTensor` in this dataset row-wise.
#'
#' @param sparse_tensor A `tf$SparseTensor`.
#'
#' @template roxlate-create-dataset
#'
#' @return A dataset of rank-(N-1) sparse tensors.
#'
#' @family tensor datasets
#'
#' @export
sparse_tensor_slices_dataset <- function(sparse_tensor, shuffle = NULL, batch_size = NULL, repeated = NULL) {

  dataset <- tf$contrib$data$Dataset$from_sparse_tensor_slices(sparse_tensor = sparse_tensor)

  if (!missing(shuffle))
    dataset <- dataset_shuffle(dataset, buffer_size = shuffle)
  if (!missing(repeated))
    dataset <- dataset_repeat(dataset, count = repeated)
  if (!missing(batch_size))
    dataset <- dataset_batch(dataset, batch_size)

  dataset
}

r_data_frame_dataset <- function() {

}

r_matrix_dataset <- function() {

}


# for data frame and matrix: https://www.tensorflow.org/programmers_guide/datasets#consuming_numpy_arrays





