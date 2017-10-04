context("tensor datasets")

source("utils.R")

test_succeeds("dataset_from_tensors produces a dataset", {
  skip_if_no_tensorflow()
  dataset_from_tensors(tf$constant(1:100))
})

test_succeeds("dataset_from_tensor_slices produces a dataset", {
  skip_if_no_tensorflow()
  dataset_from_tensor_slices(tf$constant(1:100))
})

test_succeeds("dataset_from_sparse_tensor_slices produces a dataset", {
  skip_if_no_tensorflow()
  dataset_from_sparse_tensor_slices(tf$SparseTensor(
    indices = list(c(0L, 0L), c(1L, 2L)),
    values = c(1L, 2L),
    dense_shape = c(3L, 4L)
  ))
})


