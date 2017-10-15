context("iterators")

source("utils.R")

test_succeeds("with_dataset_iterator catches end of iteration", {

  sess <- tf$Session()
  on.exit(sess$close(), add = TRUE)
  dataset <- tensor_slices_dataset(1:50) %>%
    dataset_batch(10)
  batch <- batch_from_dataset(dataset)

  with_dataset_iterator({
    while(TRUE) {
      value <- sess$run(batch)
      print(value)
    }
  })

})










