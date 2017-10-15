

dataset_prepare <- function(dataset, x, y = NULL,
                            named = TRUE, named_features = FALSE) {

}


batch_tensor
input_fn.dataset

batch_from_dataset(dataset) (make an iterator and get it's tensor)

separate iterator functions as necessary
  - one_shot_iterator()
  - iterator_get_next()
  
  
iterator_get_next() could also take a dataset and do a one_shot iterator by default?
