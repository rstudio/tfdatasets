context("range dataset")

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


test_succeeds("random_integer_dataset creates a dataset", {
    ds1 <- random_integer_dataset(seed=4L) %>% dataset_take(10)
    ds2 <- random_integer_dataset(seed=4L) %>% dataset_take(10)
    # TODO: reticulate::iterate simplify doesn't work on objects with class(x)=="numeric"
    r1 <- reticulate::iterate(ds1, as.numeric) %>% unlist()
    r2 <- reticulate::iterate(ds2, as.numeric) %>% unlist()
    expect_equal(r1, r2)
})
