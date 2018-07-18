
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

test_succeeds("dataset_shuffle_and_repeat returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_shuffle_and_repeat(20)
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

test_succeeds("dataset_skip returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_skip(1)
})

test_succeeds("dataset_map handles threads correctly and returns a dataset", {
  # force a gc within the function to ensure that these "functions" are not
  # actually called on the background thread but rather called with a placeholder
  # to yield a TF tensor which is used later.
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_map(function(x) { gc(); tf$negative(x) }, num_parallel_calls = 8) %>%
    dataset_prefetch(1)
})

test_succeeds("dataset_map_and_batch returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_map_and_batch(batch_size = 32, drop_remainder = TRUE,
      function(x) { x }
    ) %>%
    dataset_prefetch(1)
})


test_succeeds("dataset_prefetch_to_device returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_map_and_batch(batch_size = 32, drop_remainder = TRUE,
                          function(x) { x }
    ) %>%
    dataset_prefetch_to_device("/cpu:0", 1)
})

test_succeeds("dataset_filter narrows the dataset", {
  dataset <- csv_dataset("data/mtcars.csv") %>%
    dataset_filter(function(record) {
      record$mpg >= 20 & record$cyl >= 6L
    }) %>%
    dataset_batch(1000)

  sess <- tf$Session()
  on.exit(sess$close(), add = TRUE)
  batch <- next_batch(dataset)
  expect_length(sess$run(batch)$mpg, 3)
})

test_succeeds("dataset_interleave yields a dataset" , {
  dataset <- tensor_slices_dataset(c(1,2,3,4,5)) %>%
    dataset_interleave(cycle_length = 2, block_length = 4, function(x) {
      tensors_dataset(x) %>%
        dataset_repeat(6)
    })
})

test_succeeds("dataset_shard yields a dataset" , {

  dataset <- csv_dataset("data/mtcars.csv") %>%
    dataset_shard(num_shards = 4, index = 1) %>%
    dataset_batch(8)

  sess <- tf$Session()
  on.exit(sess$close(), add = TRUE)
  batch <- next_batch(dataset)
  expect_length(sess$run(batch)$mpg, 8)

})

test_succeeds("zip_datasets returns a dataset", {
  zip_datasets(list(tensors_dataset(tf$constant(1:100)), tensors_dataset(tf$constant(101:200))))
})






