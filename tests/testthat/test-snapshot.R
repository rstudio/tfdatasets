test_that("dataset_snapshot works", {

  fi <- tempfile()

  res <- tensor_slices_dataset(1:10) %>%
    dataset_snapshot(fi) %>%
    reticulate::as_iterator() %>%
    reticulate::iterate()

  expect_length(res, 10)
  expect(length(dir(fi)) > 0, "snapshots not found")
})
