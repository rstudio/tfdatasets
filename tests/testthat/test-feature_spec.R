context("feature_specs")

# Set up ------------------------------------------------------------------

source("utils.R")

skip_if_not_eager_and_tf <- function() {
  skip_if_no_tensorflow(required_version = "1.13.1")
  skip_if_not_eager()
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
  example <- reticulate::iter_next(reticulate::as_iterator(df))
  k <- keras::layer_dense_features(feature_columns = feature_columns)
  k(example)
}

# Tests -------------------------------------------------------------------


test_that("Can create a feature_spec", {
  skip_if_not_eager_and_tf()
  spec <- feature_spec(dataset, y ~ a+b+c+d)
  expect_equal(sort(spec$feature_names()), sort(names(df)[-which(names(df) == "y")]))
})

test_that("Can create numeric columns", {
  skip_if_not_eager_and_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b, c)

  spec$fit() #TODO use the fit S3 method when available

  expect_length(spec$features(), 2)
  expect_named(spec$features(), c("b", "c"))
  expect_s3_class(spec$features()[[1]], "tensorflow.python.feature_column.feature_column._DenseColumn")
  expect_s3_class(spec$features()[[2]], "tensorflow.python.feature_column.feature_column._DenseColumn")
})

test_that("Can create categorical columns with vocabulary list", {
  skip_if_not_eager_and_tf()

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
  skip_if_not_eager_and_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_categorical_column_with_hash_bucket(a, d, hash_bucket_size = 10)

  spec$fit()

  expect_length(spec$features(), 2)
  expect_named(spec$features(), c("a", "d"))
  expect_s3_class(spec$features()[[1]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")
  expect_s3_class(spec$features()[[2]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")
})

test_that("Can create categorical columns with identity", {
  skip_if_not_eager_and_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_categorical_column_with_identity(a, num_buckets = 10)

  spec$fit()

  expect_length(spec$features(), 1)
  expect_named(spec$features(), c("a"))
  expect_s3_class(spec$features()[[1]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")
})

test_that("Can create categorical columns with vocabulary file", {
  skip_if_not_eager_and_tf()

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
  skip_if_not_eager_and_tf()

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
  skip_if_not_eager_and_tf()

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
})

test_that("Can create crossed columns", {
  skip_if_not_eager_and_tf()

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
  skip_if_not_eager_and_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b) %>%
    step_bucketized_column(b, boundaries = c(5, 10, 15))

  spec$fit()

  expect_s3_class(spec$dense_features()$bucketized_b, "tensorflow.python.feature_column.feature_column_v2.BucketizedColumn")
})

test_that("Can remove columns", {
  skip_if_not_eager_and_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b) %>%
    step_bucketized_column(b, boundaries = c(5, 10, 15)) %>%
    step_remove_column(b)

  spec$fit()

  expect_length(spec$features(), 1)
})

test_that("Using with layer_dense_features", {
  skip_if_not_eager_and_tf()

  spec <- feature_spec(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b, c) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  spec$fit()

  lyr <- keras::layer_dense_features(feature_columns = spec$dense_features())

  ds <- reticulate::as_iterator(dataset)
  x <- lyr(reticulate::iter_next(ds))

  expect_equal(x$shape$as_list(), c(2, 2 + 2*26))
})

test_that("Recipes are correctly cloned/imutable", {
  skip_if_not_eager_and_tf()

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
  skip_if_not_eager_and_tf()

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
  skip_if_not_eager_and_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(b) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  spec_prep <- fit(spec)

  expect_error(dataset_use_spec(dataset, spec))
  expect_s3_class(dataset_use_spec(dataset, spec_prep), "tensorflow.python.data.ops.dataset_ops.DatasetV2")
})

test_that("Prep with different dataset", {
  skip_if_not_eager_and_tf()

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
  skip_if_not_eager_and_tf()

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
  skip_if_not_eager_and_tf()

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
  skip_if_not_eager_and_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(all_numeric(), normalizer_fn = scaler_standard())

  spec <- fit(spec)

  value <- as.matrix(get_features(dataset, spec$dense_features()))
  normalized_c <- (df$c - mean(df$c))/sd(df$c)
  normalized_b <- (df$b - mean(df$b))/sd(df$b)
  expect_equal(as.numeric(value[,2]), normalized_c[1:2], tol = 1e-6)
  expect_equal(as.numeric(value[,1]), normalized_b[1:2], tol = 1e-6)
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
  skip_if_not_eager_and_tf()

  spec <- feature_spec(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(all_numeric(), normalizer_fn = scaler_min_max())

  spec <- fit(spec)

  value <- as.matrix(get_features(dataset, spec$dense_features()))
  normalized_c <- (df$c - min(df$c))/(max(df$c) - min(df$c))
  normalized_b <- (df$b - min(df$b))/(max(df$b) - min(df$b))
  expect_equal(as.numeric(value[,2]), normalized_c[1:2], tol = 1e-6)
  expect_equal(as.numeric(value[,1]), normalized_b[1:2], tol = 1e-6)
})

