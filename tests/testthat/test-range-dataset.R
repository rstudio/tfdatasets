context("range dataset")

source("utils.R")

test_succeeds("range_dataset creates a dataset", {

  dataset <- range_dataset(from = 1, to = 11) %>% dataset_batch(10)
  batch <- next_batch(dataset)

  res <- if (tf$executing_eagerly()) {
    as.array(batch)
  } else {
    with_session(function (sess) {
      sess$run(batch)
    })
  }

  expect_equal(res, array(1L:10L))

})









