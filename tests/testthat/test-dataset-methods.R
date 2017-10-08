
context("dataset methods")

source("utils.R")

test_succeeds("dataset_repeat returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_repeat(4)
})


test_succeeds("dataset_shuffle returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_shuffle(20)
})

test_succeeds("dataset_batch returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_batch(10)
})

test_succeeds("dataset methods can be chained", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_shuffle(20) %>%
    dataset_batch(10) %>%
    dataset_repeat(4)
})

test_succeeds("dataset_cache to memory returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_cache()
})

test_succeeds("dataset_cache to disk returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_cache(filename = tempfile())
})

test_succeeds("dataset_concatenate returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_concatenate(tensors_dataset(tf$constant(1:100)))
})

test_succeeds("dataset_take returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_take(50)
})

test_succeeds("dataset_unbatch returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_batch(10) %>%
    dataset_unbatch()
})

test_succeeds("dataset_enumerate returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_enumerate()
})

test_succeeds("dataset_stip returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_skip(1)
})

test_succeeds("dataset_map handles threads correctly and returns a dataset", {
  # force a gc within the function to ensure that these "functions" are not
  # actually called on the background thread but rather called with a placeholder
  # to yield a TF tensor which is used later.
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_map(function(x) { gc(); tf$negative(x) }, num_threads = 8)
})

test_succeeds("dataset_ignore_errors ignores errors", {
  dataset <- tensor_slices_dataset(list(1, 2, 0, 4)) %>%
    dataset_map(function(x) {
      tf$check_numerics(1 / x, "error")
    }) %>%
    dataset_ignore_errors()
})


test_succeeds("zip_datasets returns a dataset", {
  zip_datasets(list(tensors_dataset(tf$constant(1:100)), tensors_dataset(tf$constant(101:200))))
})



