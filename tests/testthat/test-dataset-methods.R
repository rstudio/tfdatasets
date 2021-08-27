
context("dataset methods")


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

  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_map(~tf$negative(.x), num_parallel_calls = 8) %>%
    dataset_prefetch(1)

  if (tensorflow::tf_version() >= "2.0") {
    expect_equal(
      as.numeric(reticulate::iter_next(reticulate::as_iterator(dataset))),
      -(1:100)
    )
  }
})

test_succeeds("dataset_map_and_batch returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_map_and_batch(batch_size = 32, drop_remainder = TRUE,
      function(x) { x }
    ) %>%
    dataset_prefetch(1)

  dataset <- range_dataset(1, 100) %>%
    dataset_map_and_batch(batch_size = 32, drop_remainder = TRUE, ~ -.x) %>%
    dataset_prefetch(1)

  if (tensorflow::tf_version() >= "2.0") {
    expect_equal(
      as.numeric(reticulate::iter_next(reticulate::as_iterator(dataset))),
      -(1:32)
    )
  }
})


test_succeeds("dataset_prefetch_to_device returns a dataset", {
  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_map_and_batch(batch_size = 32, drop_remainder = TRUE,
                          function(x) { x }
    ) %>%
    dataset_prefetch_to_device("/cpu:0", 1)


  dataset <- tensors_dataset(tf$constant(1:100)) %>%
    dataset_map_and_batch(batch_size = 32, drop_remainder = TRUE,
                          function(x) { x }
    ) %>%
    dataset_prefetch_to_device("/cpu:0", 1)
})

test_succeeds("dataset_filter narrows the dataset", {
  dataset <- csv_dataset(testing_data_filepath("mtcars.csv")) %>%
    dataset_filter(function(record) {
      record$mpg >= 20 & record$cyl >= 6L
    }) %>%
    dataset_batch(1000)

  batch <- next_batch(dataset)

  res <- if (tf$executing_eagerly()) {
    batch$mpg
  } else {
    with_session(function (sess) {
      sess$run(batch)$mpg
    })
  }

  expect_length(res, 3)
})

test_succeeds("dataset_interleave yields a dataset" , {
  dataset <- tensor_slices_dataset(c(1,2,3,4,5)) %>%
    dataset_interleave(cycle_length = 2, block_length = 4, function(x) {
      tensors_dataset(x) %>%
        dataset_repeat(6)
    })
})

test_succeeds("dataset_shard yields a dataset" , {

  dataset <- csv_dataset(testing_data_filepath("mtcars.csv")) %>%
    dataset_shard(num_shards = 4, index = 1) %>%
    dataset_batch(8)

  batch <- next_batch(dataset)

  res <- if (tf$executing_eagerly()) {
    batch$mpg
  } else {
    with_session(function (sess) {
      sess$run(batch)$mpg
    })
  }

  expect_length(res, 8)

})

test_succeeds("dataset_padded_batch returns a dataset", {

  dataset <- tensor_slices_dataset(matrix(1.1:8.1, ncol = 2)) %>%
    dataset_padded_batch(
      batch_size = 2,
      padded_shapes = tf$constant(3L, shape = shape(1L), dtype = tf$int32),
      padding_values = tf$constant(77.1, dtype = tf$float64))

  dataset <- tensor_slices_dataset(matrix(1:8, ncol = 2)) %>%
    dataset_padded_batch(
      batch_size = 2,
      padded_shapes = tf$constant(3L, shape = shape(1L), dtype = tf$int32),
      padding_values = tf$constant(77L))


})

test_succeeds("zip_datasets returns a dataset", {
  zip_datasets(list(tensors_dataset(tf$constant(1:100)), tensors_dataset(tf$constant(101:200))))
})

test_succeeds("dataset_windows combines input elements into a dataset of windows", {
  d <- range_dataset(1, 100) %>%
    dataset_window(size = 5)
})

test_succeeds("dataset_collect works", {

  if (tensorflow::tf_version() < "2.0")
    skip("dataset_collect requires tf 2.0")

  dataset <- tensor_slices_dataset(1:100)

  expect_length(dataset_collect(dataset), 100)
  expect_length(dataset_collect(dataset, 1), 1)
  expect_length(dataset_collect(dataset, 10), 10)

})

test_succeeds("dataset_reduce works", {

  if (tensorflow::tf_version() < "2.0")
    skip("dataset_reduce requires tf 2.0")

  d <- tensor_slices_dataset(tf$constant(c(1.1, 2.2, 3.3)))
  sum_and_count <- d %>% dataset_reduce(tuple(0, 0), function(x, y) tuple(x[[1]] + y, x[[2]] + 1))
  expect_equal(as.numeric(sum_and_count[[1]])/as.numeric(sum_and_count[[2]]), 2.2, tolerance = 1e-6)

})


test_succeeds("length.tf_dataset", {
  expect_equal(length(range_dataset(0, 42)),
               42)

  expect_equal(length(range_dataset(0, 42) %>% dataset_repeat()),
               Inf)

  l <- range_dataset(0, 42) %>% dataset_repeat() %>%
    dataset_filter(function(x) TRUE) %>% length()
  expect_length(l, 1)
  expect_true(is.na(l))

})


test_succeeds("dataset_enumerate", {

  dataset <- tensor_slices_dataset(100:103) %>%
    dataset_enumerate()

  as_iterator <- reticulate::as_iterator
  iter_next <- reticulate::iter_next

  it <- as_iterator(dataset)
  expect_equal(iter_next(it), list(as_tensor(0, "int64"), as_tensor(100, "int64")))
  expect_equal(iter_next(it), list(as_tensor(1, "int64"), as_tensor(101, "int64")))
  expect_equal(iter_next(it), list(as_tensor(2, "int64"), as_tensor(102, "int64")))
  expect_equal(iter_next(it), list(as_tensor(3, "int64"), as_tensor(103, "int64")))
  expect_null(iter_next(it))
  expect_null(iter_next(it))

})



test_succeeds("dataset_scan", {
  initial_state <- as_tensor(0, dtype = "int64")
  scan_func <- function(state, i) list(state + i, state + i)
  dataset <- range_dataset(0, 10) %>%
    dataset_scan(initial_state, scan_func)

  res <- reticulate::iterate(dataset, as.array) %>%
    unlist()
  expect_equal(res, c(0, 1, 3, 6, 10, 15, 21, 28, 36, 45))

})
