
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

