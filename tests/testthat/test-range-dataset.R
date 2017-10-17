context("range dataset")

source("utils.R")

test_succeeds("range_dataset creates a dataset", {
  sess <- tf$Session()
  on.exit(sess$close(), add = TRUE)
  expect_equal(
    sess$run(next_batch(range_dataset(from = 1, to = 11) %>% dataset_batch(10))),
    array(1L:10L)
  )
})









