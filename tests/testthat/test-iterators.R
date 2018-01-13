context("iterators")

source("utils.R")

test_that("make_iterator_one_shot works", {

  skip_if_no_tensorflow()

  batch <- mtcars_dataset() %>%
    make_iterator_one_shot() %>%
    iterator_get_next()

  with_session(function(sess) {
    records <- sess$run(batch)
    expect_type(records$disp, "double")
  })
})

test_that("make_iterator_initializable works", {

  skip_if_no_tensorflow()

  with_session(function(sess) {

    max_value <- tf$placeholder(tf$int64, shape = shape())
    range_ds <- range_dataset(from = 1, to = max_value)

    iterator <- range_ds %>%
      make_iterator_initializable()

    next_element <- iterator_get_next(iterator)

    iterator %>%
      iterator_initializer() %>%
      sess$run(feed_dict = dict(max_value = 10L))

    for (i in 1L:9L) {
      value <- sess$run(next_element)
      expect_equal(i, value)
    }

    iterator %>%
      iterator_initializer() %>%
      sess$run(feed_dict = dict(max_value = 20L))

    for (i in 1L:19L) {
      value <- sess$run(next_element)
      expect_equal(i, value)
    }
  })

})

test_succeeds("make_iterator_from_structure works", {

  skip_if_no_tensorflow()

  with_session(function(sess) {

    training_dataset <- range_dataset(from = 1, to = 100) %>%
      dataset_map(function(x) {
        x + tf$random_uniform(shape(), -10L, 10L, tf$int64)
      })

    validation_dataset = range_dataset(from = 1, to = 50)

    iterator <- make_iterator_from_structure(output_types(training_dataset),
                                             output_shapes(training_dataset))

    next_element <- iterator_get_next(iterator)

    training_init_op <- iterator_make_initializer(iterator, training_dataset)
    validation_init_op = iterator_make_initializer(iterator, validation_dataset)

    for (i in 1:20) {

      # Initialize an iterator over the training dataset.
      sess$run(training_init_op)
      for (j in 1:99)
        sess$run(next_element)

      # Initialize an iterator over the validation dataset.
      sess$run(validation_init_op)
      for (j in 1:49)
        sess$run(next_element)
    }
  })
})

test_succeeds("make_iterator_from_string_handle works", {

  skip_if_no_tensorflow()

  with_session(function(sess) {

    training_dataset <- range_dataset(from = 1, to = 100) %>%
      dataset_map(function(x) {
        x + tf$random_uniform(shape(), -10L, 10L, tf$int64)
      }) %>%
      dataset_repeat()

    validation_dataset = range_dataset(from = 1, to = 50)

    handle <- tf$placeholder(tf$string, shape = shape())

    iterator <- make_iterator_from_string_handle(
      handle,
      output_types(training_dataset),
      output_shapes(training_dataset)
    )

    next_element <- iterator_get_next(iterator)

    training_iterator <- make_iterator_one_shot(training_dataset)
    validation_iterator <- make_iterator_initializable(validation_dataset)

    training_handle <- sess$run(iterator_string_handle(training_iterator))
    validation_handle <- sess$run(iterator_string_handle(validation_iterator))

    for (i in 1:2) {

      for (j in 1:199)
        sess$run(next_element, feed_dict = dict(handle = training_handle))

      sess$run(iterator_initializer(validation_iterator))
      for (j in 1:49)
        sess$run(next_element, feed_dict = dict(handle = validation_handle))
    }

  })

})


