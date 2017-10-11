
context("dataset iterators")

source("utils.R")

test_succeeds("iterator_from_dataset returns an iterator", {
  dataset <- tensors_dataset(tf$constant(1:100))
  iterator_from_dataset(dataset)
})


test_succeeds("next_element returns values", {
  x <- 1:100
  dataset <- tensors_dataset(tf$constant(x))
  iter <- iterator_from_dataset(dataset)
  sess <- tf$Session()
  on.exit(sess$close(), add = TRUE)
  next_element(iter, sess)
})

