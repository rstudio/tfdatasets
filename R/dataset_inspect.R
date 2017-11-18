


dataset_has_class <- function(dataset, class) {
  input_dataset <- dataset
  while (!is.null(input_dataset)) {
    if (inherits(input_dataset, class))
      return(TRUE)
    input_dataset <- if (py_has_attr(input_dataset, "_input_dataset"))
      input_dataset$`_input_dataset`
    else
      NULL
  }
  FALSE
}

dataset_is_batched <- function(dataset) {
  dataset_has_class(dataset, "tensorflow.python.data.ops.dataset_ops.BatchDataset")
}


