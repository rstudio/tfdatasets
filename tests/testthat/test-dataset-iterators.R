
context("dataset iterators")

source("utils.R")

test_succeeds("one_shot_iterator returns an iterator", {
  dataset <- tensors_dataset(tf$constant(1:100))
  one_shot_iterator(dataset)
})

test_succeeds("initializable_iterator returns an iterator", {
  dataset <- tensors_dataset(tf$constant(1:100))
  initializable_iterator(dataset)
})

test_succeeds("iterator_next returns values", {
  x <- 1:100
  dataset <- tensors_dataset(tf$constant(x))
  iter <- one_shot_iterator(dataset)
  sess <- tf$Session()
  on.exit(sess$close(), add = TRUE)
  iterator_next(iter, sess)
})

