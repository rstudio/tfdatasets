library(tensorflow)
library(tfdatasets)
library(tfestimators)
library(keras)
library(tidyverse)

URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
df <- read_csv(URL)

to_dataset <-  . %>%
  tensor_slices_dataset() %>%
  dataset_prepare(x = -target, y = target, named_features = TRUE, named = FALSE) %>%
  #dataset_shuffle(nrow(df)) %>%
  dataset_batch(32)

id_train <- sample.int(nrow(df), size = 0.7*nrow(df))
train_dataset <- to_dataset(df[id_train,])
valid_dataset <- to_dataset(df[-id_train,])

# What does each column helper does ---------------------------------------

example_batch <- train_dataset %>%
  reticulate::as_iterator() %>%
  reticulate::iter_next()

demo <- function(x, example_batch) {
  feature_layer = keras::layer_dense_features(feature_columns = x)
  print(feature_layer(example_batch[[1]])$numpy())
}

age <- column_numeric("age")
demo(feature_columns(age = age), example_batch)

age_buckets <- column_bucketized(age, boundaries=c(18, 25, 30, 35, 40, 45, 50, 55, 60, 65))
demo(feature_columns(age = age_buckets), example_batch)

thal <- column_categorical_with_vocabulary_list("thal", vocabulary_list = unique(df$thal))
demo(feature_columns(thal = column_embedding(thal, dimension = 3)), example_batch)
demo(feature_columns(thal = column_indicator(thal)), example_batch)

crossed_feature <- column_crossed(list(age_buckets, thal), hash_bucket_size = 10)
demo(feature_columns(column_indicator(crossed_feature)), example_batch)

# Create features specification -------------------------------------------

# TODO investigate crossed column in features.
features <- feature_columns(names = df,
  c(age, trestbps, chol, thalach, oldpeak, slope, ca) ~ column_numeric(),
  column_bucketized(age, boundaries=c(18, 25, 30, 35, 40, 45, 50, 55, 60, 65)),
  column_indicator(column_categorical_with_vocabulary_list("thal", vocabulary_list = unique(df$thal))),
  column_embedding(column_categorical_with_vocabulary_list("thal", vocabulary_list = unique(df$thal)), dimension = 2)
)

feature_layer <- layer_dense_features(feature_columns = features)

feature_layer(example_batch[[1]]) # see what feature_layer does

# Model -------------------------------------------------------------------

model <- keras_model_sequential(list(
  feature_layer,
  layer_dense(units = 128, activation='relu'),
  layer_dense(units = 128, activation='relu'),
  layer_dense(units = 1, activation='sigmoid')
  ))

model %>% compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=custom_metric("acc", metric_binary_accuracy)
  )

# TODO investigate warnings after fitting.
model %>% fit(
  train_dataset,
  validation_data=valid_dataset,
  epochs=5L
  )

model %>% predict(train_dataset) %>% sort() %>%  plot()
predictions <- predict(model, train_dataset)

evaluate(model, valid_dataset)

