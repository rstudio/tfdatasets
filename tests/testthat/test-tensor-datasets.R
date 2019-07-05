context("tensor datasets")

source("utils.R")

test_succeeds("tensors_dataset creates a dataset", {
  tensors_dataset(tf$constant(1:100))
})

test_succeeds("tensor_slices_dataset create a dataset", {
  tensor_slices_dataset(tf$constant(1:100))
})

test_succeeds("sparse_tensor_slices_dataset creates a dataset", {

  skip_if_v2("from_sparse_tensor_slices is not available in TF 2.0")

  sparse_tensor_slices_dataset(tf$SparseTensor(
    indices = list(c(0L, 0L), c(1L, 2L)),
    values = c(1L, 2L),
    dense_shape = c(3L, 4L)
  ))
})

test_succeeds("tensor slices works with data.frames", {
  tensor_slices_dataset(mtcars)
})

test_succeeds("tensor slices works with unamed lists", {
  tensor_slices_dataset(list(1:3, 1:3, 1:3))
})

test_succeeds("tensor slices works with mixed named/unnamed lists", {
  # TODO is this the expected behavior?
  tensor_slices_dataset(list(1:3, a = 1:3, 1:3))
})








