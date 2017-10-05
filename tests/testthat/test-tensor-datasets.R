context("tensor datasets")

source("utils.R")

test_succeeds("tensors_dataset creates a dataset", {
  tensors_dataset(tf$constant(1:100))
})

test_succeeds("tensor_slices_dataset create a dataset", {
  tensor_slices_dataset(tf$constant(1:100))
})

test_succeeds("sparse_tensor_slices_dataset creates a dataset", {
  sparse_tensor_slices_dataset(tf$SparseTensor(
    indices = list(c(0L, 0L), c(1L, 2L)),
    values = c(1L, 2L),
    dense_shape = c(3L, 4L)
  ))
})









