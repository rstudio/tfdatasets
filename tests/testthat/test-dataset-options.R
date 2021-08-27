test_that("dataset options", {

  skip_if(tf_version() <= "2.5")
  # literally every option is experimental before 2.5, no stable api


  ds <- range_dataset(0, 10)

  expect_null(ds$options()$experimental_deterministic)
  expect_null(ds$options()$threading$private_threadpool_size)

  ds1 <- ds %>%
    dataset_options(
      experimental_deterministic = FALSE,
      threading.private_threadpool_size = 10
    )

  expect_false(ds1$options()$experimental_deterministic)
  expect_equal(ds1$options()$threading$private_threadpool_size, 10)

  # pass options as a named list:
  opts <- list(
    experimental_deterministic = FALSE,
    threading.private_threadpool_size = 10
  )
  ds1 <- range_dataset(0, 10) %>% dataset_options(opts)

  expect_false(ds1$options()$experimental_deterministic)
  expect_equal(ds1$options()$threading$private_threadpool_size, 10)


  # pass a tf.data.Options() instance
  opts <- tf$data$Options()
  opts$experimental_deterministic <- FALSE
  opts$threading$private_threadpool_size <- 10L
  ds1 <- range_dataset(0, 10) %>% dataset_options(opts)


  expect_false(ds1$options()$experimental_deterministic)
  expect_equal(ds1$options()$threading$private_threadpool_size, 10)

  # get currently set options
  o1 <- range_dataset(0, 10)$options()
  o2 <- range_dataset(0, 10) %>% dataset_options()
  expect_equal(class(o1), class(o2))
  expect_equal(o1, o2)

})


