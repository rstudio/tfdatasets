context("keras-input-tensor")

library(keras)
K <- backend()
K$get_session()

source("utils.R")

# TODO:
#  - iterator_from_dataset should return the tensor (rename to e.g. records_from_dataset?)
#
#  - get rid of next_element and next_element_tensor (since both estimators and keras will
#    already take care of evaluating the tensor in the right session)
#
#  - named_features should be FALSE by default and input_fn will pass TRUE as necessary
#
#  - is there a way to repeat "forever"
#
#  - extend example to save and load weights
#


test_succeeds("iterators can provide keras input tensors", {

  # create dataset
  dataset <- csv_dataset("data/iris.csv") %>%
    dataset_map(function(record) {
      record$Species <- tf$one_hot(record$Species, depth = 3L)
      record
    }) %>%
    dataset_shuffle(50) %>%
    dataset_batch(10) %>%
    dataset_repeat(5)

  # stream batches from dataset
  train_batch <- batch_from_dataset(dataset, features = -Species, response = Species)

  # create model
  input <- layer_input(tensor = train_batch[[1]], batch_shape = shape(NULL, 4))
  predictions <- input %>%
    layer_dense(units = 10, activation = "relu") %>%
    layer_dense(units = 20, activation = "relu") %>%
    layer_dense(units = 3, activation = "softmax")
  model <- keras_model(input, predictions)
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy'),
    target_tensors = train_batch[[2]]
  )

  # fit with the generator
  model %>% fit(
    steps_per_epoch = 15,
    epochs = 5
  )



})











