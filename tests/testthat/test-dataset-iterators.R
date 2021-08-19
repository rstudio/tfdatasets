context("iterators")

test_succeeds("with_dataset catches end of iteration", {

  dataset <- tensor_slices_dataset(1:50) %>%
    dataset_batch(10)

  if (tf$executing_eagerly()) {
    iter <- make_iterator_one_shot(dataset)
    with_dataset({
      while (TRUE) {
        batch <- iterator_get_next(iter)
        print(batch)
      }
    })
  } else {
    batch <- next_batch(dataset)
    with_session(function (sess) {
      with_dataset({
        while (TRUE) {
          value <- sess$run(batch)
          print(value)
        }
      })
    })
  }
})

test_succeeds("until_out_of_range catches end of iteration", {

  dataset <- tensor_slices_dataset(1:50) %>%
    dataset_batch(10)

  if (tf$executing_eagerly()) {
    iter <- make_iterator_one_shot(dataset)
    until_out_of_range({
        batch <- iterator_get_next(iter)
        print(batch)
    })
  } else {
    batch <- next_batch(dataset)
    with_session(function (sess) {
      until_out_of_range({
          value <- sess$run(batch)
          print(value)
      })
    })
  }
})

test_succeeds("until_out_of_range catches break", {
  until_out_of_range({
    break
  })
})
