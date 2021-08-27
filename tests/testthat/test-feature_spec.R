context("feature_specs")

# Set up ------------------------------------------------------------------

skip_if_not_tf <- function() {
  skip_if_no_tensorflow(required_version = "2.0")
}

df <- list(
  a = letters,
  b = 1:length(letters),
  c = runif(length(letters)),
  d = LETTERS,
  y = runif(length(letters))
)

dataset <-  df %>%
  tensor_slices_dataset() %>%
  dataset_batch(2)

get_features <- function(df, feature_columns) {

  if (tensorflow::tf$executing_eagerly())
    example <- reticulate::iter_next(reticulate::as_iterator(df))
  else {
    example <- make_iterator_one_shot(df)
    example <- iterator_get_next(example)
  }

  k <- keras::layer_dense_features(feature_columns = feature_columns)

  if (tensorflow::tf$executing_eagerly())
    return(k(example))
  else {
    res <- k(example)
    sess <- tf$Session()
    return(sess$run(res))
  }
}

# Tests -------------------------------------------------------------------


test_that("Can create a feature_spec", {
  skip_if_not_tf()
  spec <- feature_spec(dataset, y ~ a+b+c+d)
  expect_equal(sort(spec$feature_names()), sort(names(df)[-which(names(df) == "y")]))
})

test_that("Can create numeric columns", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b, c)

  spec$fit() #TODO use the fit S3 method when available

  expect_length(spec$features(), 2)
  expect_named(spec$features(), c("b", "c"))
  expect_s3_class(spec$features()[[1]], "tensorflow.python.feature_column.feature_column._DenseColumn")
  expect_s3_class(spec$features()[[2]], "tensorflow.python.feature_column.feature_column._DenseColumn")
})

test_that("Can create categorical columns with vocabulary list", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_categorical_column_with_vocabulary_list(a, d)

  spec$fit()

  expect_length(spec$features(), 2)
  expect_named(spec$features(), c("a", "d"))
  expect_s3_class(spec$features()[[1]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")
  expect_s3_class(spec$features()[[2]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, vocabulary_list = letters[1:5])

  spec$fit()
  expect_length(spec$features(), 1)
  expect_length(spec$dense_features(), 0)
})

test_that("Can create categorical columns with hash_bucket", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_categorical_column_with_hash_bucket(a, d, hash_bucket_size = 10)

  spec$fit()

  expect_length(spec$features(), 2)
  expect_named(spec$features(), c("a", "d"))
  expect_s3_class(spec$features()[[1]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")
  expect_s3_class(spec$features()[[2]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")
})

test_that("Can create categorical columns with identity", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_categorical_column_with_identity(a, num_buckets = 10)

  spec$fit()

  expect_length(spec$features(), 1)
  expect_named(spec$features(), c("a"))
  expect_s3_class(spec$features()[[1]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")
})

test_that("Can create categorical columns with vocabulary file", {
  skip_if_not_tf()

  tmp <- tempfile()
  writeLines(tmp, text = letters)

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_categorical_column_with_vocabulary_file(a, vocabulary_file = tmp)

  spec$fit()

  expect_length(spec$features(), 1)
  expect_named(spec$features(), c("a"))
  expect_s3_class(spec$features()[[1]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")
})

test_that("Can create indicator variables", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  spec$fit()

  expect_length(spec$dense_features(), 2)
  expect_named(spec$dense_features(), c("indicator_a", "indicator_d"))
  expect_s3_class(spec$dense_features()[[1]], "tensorflow.python.feature_column.feature_column_v2.IndicatorColumn")

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(ind_a = a)

  spec$fit()

  expect_named(spec$dense_features(), c("ind_a"))
})

test_that("Can create embedding columns", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_embedding_column(a, d, dimension = 5)

  spec$fit()

  expect_length(spec$dense_features(), 2)
  expect_named(spec$dense_features(), c("embedding_a", "embedding_d"))
  expect_s3_class(spec$dense_features()[[1]], "tensorflow.python.feature_column.feature_column_v2.EmbeddingColumn")

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_embedding_column(emb_a = a, dimension = 5)

  spec$fit()

  expect_named(spec$dense_features(), c("emb_a"))

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_embedding_column(a, d)

  spec$fit()

  expect_length(spec$dense_features(), 2)
  expect_named(spec$dense_features(), c("embedding_a", "embedding_d"))
  expect_s3_class(spec$dense_features()[[1]], "tensorflow.python.feature_column.feature_column_v2.EmbeddingColumn")
})



test_that("Can create crossed columns", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_crossed_column(c(a, d), hash_bucket_size = 100) %>%
    step_indicator_column(crossed_a_d)

  spec$fit()


  expect_named(spec$dense_features(), "indicator_crossed_a_d")
  expect_s3_class(spec$dense_features()[[1]], "tensorflow.python.feature_column.feature_column_v2.IndicatorColumn")
  expect_s3_class(spec$features()$crossed_a_d, "tensorflow.python.feature_column.feature_column_v2.CrossedColumn")
})

test_that("Can create bucketized columns", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b) %>%
    step_bucketized_column(b, boundaries = c(5, 10, 15))

  spec$fit()

  expect_s3_class(spec$dense_features()$bucketized_b, "tensorflow.python.feature_column.feature_column_v2.BucketizedColumn")
})

test_that("Can remove columns", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b) %>%
    step_bucketized_column(b, boundaries = c(5, 10, 15)) %>%
    step_remove_column(b)

  spec$fit()

  expect_length(spec$features(), 1)
})

test_that("Using with layer_dense_features", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b, c) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  spec$fit()

  lyr <- keras::layer_dense_features(feature_columns = spec$dense_features())

  ds <- reticulate::as_iterator(dataset)
  x <- lyr(reticulate::iter_next(ds))

  if (tensorflow::tf$executing_eagerly())
    expect_equal(x$shape$as_list(), c(2, 2 + 2*26))
  else
    expect_equal(x$shape$as_list()[[2]], 2 + 2*26)
})

test_that("Recipes are correctly cloned/imutable", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b, c) %>%
    step_categorical_column_with_vocabulary_list(a, d)

  spec1 <- spec %>%
    step_indicator_column(a, d)

  spec2 <- spec %>%
    step_indicator_column(a, d)

  spec1$fit()

  expect_length(spec1$features(), 6)
  expect_error(spec2$features())
  expect_error(spec$features())

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b, c) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  spec_prep <- fit(spec)

  expect_length(spec_prep$features(), 6)
  expect_error(spec$features())
})


test_that("Recipes column types", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  expect_equal(
    spec$feature_types(),
    c("float32", "string", "string", "float32", "float32", "float32")
  )
})

test_that("Fit feature_spec", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(b) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  spec_prep <- fit(spec)

  expect_error(dataset_use_spec(dataset, spec))
  expect_s3_class(dataset_use_spec(dataset, spec_prep), "tensorflow.python.data.ops.dataset_ops.DatasetV2")
})

test_that("Prep with different dataset", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(b) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  ds <- df %>%
    tensor_slices_dataset() %>%
    dataset_take(10)

  spec_prep <- fit(spec, ds)

  expect_s3_class(dataset_use_spec(ds, spec_prep), "tensorflow.python.data.ops.dataset_ops.DatasetV2")
})

test_that("Can select with has_type", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(has_type("float32")) %>%
    step_numeric_column(has_type("int32"))

  expect_length(spec$steps, 2)

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(has_type("float32")) %>%
    step_numeric_column(has_type("int32")) %>%
    step_categorical_column_with_vocabulary_list(has_type("string")) %>%
    step_indicator_column(has_type("string"))

  expect_length(spec$steps, 6)
  expect_error(spec %>% step_indicator_column(a = has_type("string")))

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(all_numeric()) %>%
    step_categorical_column_with_vocabulary_list(has_type("string")) %>%
    step_indicator_column(all_nominal())

  expect_length(spec$steps, 6)
})

test_that("Can remove variables using -", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(all_numeric(), - b) %>%
    step_categorical_column_with_vocabulary_list(all_nominal()) %>%
    step_indicator_column(all_nominal(), - a)

  spec <- fit(spec)

  expect_length(spec$dense_features(), 2)
  expect_named(spec$dense_features(), c("c", "indicator_d"))
})

test_that("StandardScaler works as expected", {
  x <- runif(100)
  sc <- StandardScaler$new()
  splited <- split(x, rep(1:10, each = 10))
  a <- lapply(splited, sc$fit_batch)
  sc$fit_resume()

  expect_equal(sc$mean, mean(x))
  expect_equal(sc$sd, sd(x))
})

test_that("Can use a scaler_standard", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(all_numeric(), normalizer_fn = scaler_standard())

  spec <- fit(spec)

  value <- as.matrix(get_features(dataset, spec$dense_features()))
  normalized_c <- (df$c - mean(df$c))/sd(df$c)
  normalized_b <- (df$b - mean(df$b))/sd(df$b)
  expect_equal(as.numeric(value[,2]), normalized_c[1:2], tolerance = 1e-6)
  expect_equal(as.numeric(value[,1]), normalized_b[1:2], tolerance = 1e-6)
})

test_that("MinMaxScaler works as expected", {
  x <- runif(100)
  sc <- MinMaxScaler$new()
  splited <- split(x, rep(1:10, each = 10))
  a <- lapply(splited, sc$fit_batch)
  sc$fit_resume()

  expect_equal(sc$min, min(x))
  expect_equal(sc$max, max(x))
})

test_that("Can use a scaler_min_max", {
  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(all_numeric(), normalizer_fn = scaler_min_max())

  spec <- fit(spec)

  value <- as.matrix(get_features(dataset, spec$dense_features()))
  normalized_c <- (df$c - min(df$c))/(max(df$c) - min(df$c))
  normalized_b <- (df$b - min(df$b))/(max(df$b) - min(df$b))
  expect_equal(as.numeric(value[,2]), normalized_c[1:2], tolerance = 1e-6)
  expect_equal(as.numeric(value[,1]), normalized_b[1:2], tolerance = 1e-6)
})

test_that("Can use layer_input_from_dataset with TF datasets", {

  skip_if_not_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(all_numeric(), normalizer_fn = scaler_min_max())

  spec <- fit(spec)

  ds <- dataset_use_spec(dataset, spec)
  input <- layer_input_from_dataset(ds)

  output <- input %>%
    keras::layer_dense_features(spec$dense_features())

  model <- keras::keras_model(inputs = input, outputs = output)


  expect_length(input, 4)
  if (tf$executing_eagerly())
    expect_equal(dim(as.matrix(model(next_batch(ds)[[1]]))), c(2,2))
})

test_that("Can use layer_input_from_dataset with TF data frames", {

  skip_if_not_tf()

  spec <- feature_spec(as.data.frame(df), y ~ a + b + c + d) %>%
    step_numeric_column(all_numeric(), normalizer_fn = scaler_min_max())

  spec <- fit(spec)

  input <- layer_input_from_dataset(as.data.frame(df)[, 1:4])

  output <- input %>%
    keras::layer_dense_features(spec$dense_features()) %>%
    keras::layer_dense(units = 1)

  model <- keras::keras_model(inputs = input, outputs = output)
  keras::compile(model, loss = "mse", optimizer = "adam")
  hist <- keras::fit(model, x = df, y = df$y, verbose = 0)

  expect_s3_class(hist, "keras_training_history")
})

test_that("Can use data.frames", {

  skip_if_not_tf()

  spec <- feature_spec(hearts, target ~ .) %>%
    step_numeric_column(
      all_numeric(), -cp, -restecg, -exang, -sex, -fbs,
      normalizer_fn = scaler_standard()
    ) %>%
    step_categorical_column_with_vocabulary_list(thal) %>%
    step_bucketized_column(age, boundaries = c(18, 25, 30, 35, 40, 45, 50, 55, 60, 65)) %>%
    step_indicator_column(thal) %>%
    step_embedding_column(thal, dimension = 2) %>%
    step_crossed_column(c(thal, bucketized_age), hash_bucket_size = 10) %>%
    step_indicator_column(crossed_thal_bucketized_age) %>%
    fit()

  expect_length(spec$dense_features(), 11)
})

test_that("Correctly creates indicator vars", {
  skip_if_not_tf()

  x <- data.frame(
    y = runif(5),
    x = c("a", "aã", "b", "c", "d"),
    b = runif(5),
    stringsAsFactors = FALSE
  )

  spec <- feature_spec(x, y ~ x) %>%
    step_categorical_column_with_vocabulary_list(x) %>%
    step_indicator_column(x)

  spec <- fit(spec)

  k <- keras::layer_dense_features(feature_columns = spec$dense_features())
  res <- as.matrix(k(list(x = x$x)))
  expect_equal(
    res,
    diag(nrow(res))
  )
})

test_that("feature_spec works with make_csv_dataset", {
  skip_if_not_tf()

  TRAIN_DATA_URL <- "https://storage.googleapis.com/tf-datasets/titanic/train.csv"

  train_file_path <- keras::get_file("train_csv", TRAIN_DATA_URL)
  train_dataset <- make_csv_dataset(
    train_file_path,
    field_delim = ",",
    batch_size = 5,
    num_epochs = 1
  )

  spec <- feature_spec(train_dataset, survived ~ .)

  expect_s3_class(spec, class = "FeatureSpec")
})

test_that("can create image embedding steps", {
  skip_if_not_tf()

  if (tensorflow::tf$executing_eagerly())
    skip("Needs non-eager execution.")

  df <- list(img = array(0, dim = c(1, 192, 192, 3)))
  df <- tensor_slices_dataset(df)

  spec <- feature_spec(df, x = c(img)) %>%
    step_image_embedding_column(
      img,
      module_spec = "https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/quantops/feature_vector/3"
    )

  spec <- spec %>% fit()

  layer <- keras::layer_dense_features(feature_columns = spec$dense_features())
  x <- layer(list(img = array(0, dim = c(1, 192, 192, 3))))

  expect_equal(x$get_shape()$as_list(), c(1L, 768L))
})

test_that("can create text embedding columns", {
  # TODO: this was removed in tfhub, delete this test
  skip_if_not_tf()

  if (tensorflow::tf$executing_eagerly())
    skip("Needs non-eager execution.")

  df <- list(txt = c("hello world", "hello world"))
  df <- tensor_slices_dataset(df)

  spec <- feature_spec(df, x = c(txt)) %>%
    step_text_embedding_column(txt, module_spec = "https://tfhub.dev/google/nnlm-en-dim50/1")

  spec <- spec %>% fit()

  layer <- keras::layer_dense_features(feature_columns = spec$dense_features())
  x <- layer(list(txt = c("hello world", "hello world")))

  expect_equal(x$get_shape()$as_list(), list(NULL, 50L))
})

test_that("can save and reload models that use a normalizer_fn", {

  data <- data.frame(
    y = runif(5),
    x = runif(5),
    b = runif(5)
  )

  spec <- feature_spec(data, y ~ .) %>%
    step_numeric_column(x, normalizer_fn = scaler_standard()) %>%
    fit()

  input <- layer_input_from_dataset(data[-1])
  output <- input %>%
    layer_dense_features(dense_features(spec)) %>%
    layer_dense(units = 1, activation = "sigmoid")
  model <- keras_model(input, output)

  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "binary_accuracy"
  )

  tmp <- tempfile("model")
  rds <- tempfile("rds")

  save_model_weights_tf(model, tmp)
  saveRDS(spec, rds)

  reloaded_spec <- readRDS(rds)
  input <- layer_input_from_dataset(data[-1])
  output <- input %>%
    layer_dense_features(dense_features(reloaded_spec)) %>%
    layer_dense(units = 1, activation = "sigmoid")
  new_model <- keras_model(input, output)
  load_model_weights_tf(new_model, tmp)

  expect_equal(
    predict(model, data[-1]),
    predict(new_model, data[-1])
  )

})
