

# initialize tensorflow
library(tensorflow)
use_condaenv("r-tensorflow", required = TRUE)
sess <- tf$Session()

# use estimators
library(tfestimators)

# import tensor hub
library(reticulate)
# conda_install("r-tensorflow", "tensorflow-hub", pip = TRUE)
hub <- import("tensorflow_hub")
embed <- hub$Module("https://tfhub.dev/google/universal-sentence-encoder/1")

# dataset mapping files to 1D text tensors
library(tfdatasets)
dataset <- file_list_dataset("data/*.txt") %>%
  dataset_map(num_parallel_calls = 4, function(record) {
    list(
      sentence = tf$read_file(record),
      sentiment = tf$constant(1)
    )
  }) %>%
  dataset_batch(128) %>%
  dataset_repeat(10)


embedded_text_feature_column <- hub$text_embedding_column(
  key = "sentence",
  module_spec = "https://tfhub.dev/google/universal-sentence-encoder/1"
)

estimator <- dnn_classifier(
  hidden_units= c(500, 100),
  feature_columns = embedded_text_feature_column,
  n_classes = 2,
  optimizer = "Adagrad"
)

estimator %>% train(
  input_fn(dataset, features = "sentence", response = "sentiment")
)


# call tensors directly to generate embeddings
batch <- next_batch(dataset)
embeddings <- embed(batch$sentence)
sess$run(list(tf$global_variables_initializer(), tf$tables_initializer()))
sess$run(embeddings)






