context("tensor datasets")

source("utils.R")

test_succeeds("tensors_dataset produces a dataset", {
  skip_if_no_tensorflow()
  tensors_dataset(tf$constant(1:100))
})

test_succeeds("tensor_slices_dataset produces a dataset", {
  skip_if_no_tensorflow()
  tensor_slices_dataset(tf$constant(1:100))
})

test_succeeds("sparse_tensor_slices_dataset produces a dataset", {
  skip_if_no_tensorflow()
  sparse_tensor_slices_dataset(tf$SparseTensor(
    indices = list(c(0L, 0L), c(1L, 2L)),
    values = c(1L, 2L),
    dense_shape = c(3L, 4L)
  ))
})









