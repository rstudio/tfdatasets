context("iterators")

source("utils.R")

test_succeeds("with_dataset catches end of iteration", {

  sess <- tf$Session()
  on.exit(sess$close(), add = TRUE)
  dataset <- tensor_slices_dataset(1:50) %>%
    dataset_batch(10)
  batch <- next_batch(dataset)

  with_dataset({
    while(TRUE) {
      value <- sess$run(batch)
      print(value)
    }
  })

})

test_succeeds("until_out_of_range catches end of iteration", {

  sess <- tf$Session()
  on.exit(sess$close(), add = TRUE)
  dataset <- tensor_slices_dataset(1:50) %>%
    dataset_batch(10)
  batch <- next_batch(dataset)

  until_out_of_range({
    value <- sess$run(batch)
    print(value)
  })

})


test_succeeds("until_out_of_range catches break", {
  until_out_of_range({
    break
  })
})






