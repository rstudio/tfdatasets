
context("dataset properties")

source("utils.R")

test_succeeds("output_types returns types", {
  dataset <- tensors_dataset(tf$constant(1:100))
  output_types(dataset)
})

test_succeeds("output_shapes returns shapes", {
  dataset <- tensors_dataset(tf$constant(1:100))
  output_shapes(dataset)
})

