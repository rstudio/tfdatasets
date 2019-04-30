context("recipes")

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

# Tests -------------------------------------------------------------------


test_that("Can create a recipe", {
  skip_if_not_eager_and_tf()
  rec <- recipe(dataset, y ~ a+b+c+d)
  expect_equal(sort(rec$feature_names()), sort(names(df)[-which(names(df) == "y")]))
})

test_that("Can create numeric columns", {
  skip_if_not_eager_and_tf()

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b, c)

  rec$fit() #TODO use the fit S3 method when available

  expect_length(rec$features(), 2)
  expect_named(rec$features(), c("b", "c"))
  expect_s3_class(rec$features()[[1]], "tensorflow.python.feature_column.feature_column._DenseColumn")
  expect_s3_class(rec$features()[[2]], "tensorflow.python.feature_column.feature_column._DenseColumn")
})

test_that("Can create categorical columns with vocabulary list", {
  skip_if_not_eager_and_tf()

  rec <- recipe(dataset, y ~ a + b + c + d) %>%
    step_categorical_column_with_vocabulary_list(a, d)

  rec$fit()

  expect_length(rec$features(), 2)
  expect_named(rec$features(), c("a", "d"))
  expect_s3_class(rec$features()[[1]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")
  expect_s3_class(rec$features()[[2]], "tensorflow.python.feature_column.feature_column._CategoricalColumn")

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, vocabulary_list = letters[1:5])

  rec$fit()
  expect_length(rec$features(), 1)
  expect_length(rec$dense_features(), 0)
})

test_that("Can create indicator variables", {
  skip_if_not_eager_and_tf()

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  rec$fit()

  expect_length(rec$dense_features(), 2)
  expect_named(rec$dense_features(), c("indicator_a", "indicator_d"))
  expect_s3_class(rec$dense_features()[[1]], "tensorflow.python.feature_column.feature_column_v2.IndicatorColumn")

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(ind_a = a)

  rec$fit()

  expect_named(rec$dense_features(), c("ind_a"))
})

test_that("Can create embedding columns", {
  skip_if_not_eager_and_tf()

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_embedding_column(a, d, dimension = 5)

  rec$fit()

  expect_length(rec$dense_features(), 2)
  expect_named(rec$dense_features(), c("embedding_a", "embedding_d"))
  expect_s3_class(rec$dense_features()[[1]], "tensorflow.python.feature_column.feature_column_v2.EmbeddingColumn")

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_embedding_column(emb_a = a, dimension = 5)

  rec$fit()

  expect_named(rec$dense_features(), c("emb_a"))
})

test_that("Can create crossed columns", {
  skip_if_not_eager_and_tf()

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_crossed_column(c(a, d), hash_bucket_size = 100) %>%
    step_indicator_column(crossed_a_d)

  rec$fit()


  expect_named(rec$dense_features(), "indicator_crossed_a_d")
  expect_s3_class(rec$dense_features()[[1]], "tensorflow.python.feature_column.feature_column_v2.IndicatorColumn")
  expect_s3_class(rec$features()$crossed_a_d, "tensorflow.python.feature_column.feature_column_v2.CrossedColumn")
})

test_that("Can create bucketized columns", {

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b) %>%
    step_bucketized_column(b, boundaries = c(5, 10, 15))

  rec$fit()

  expect_s3_class(rec$dense_features()$bucketized_b, "tensorflow.python.feature_column.feature_column_v2.BucketizedColumn")
})

test_that("Using with layer_dense_features", {

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b, c) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  rec$fit()

  lyr <- keras::layer_dense_features(feature_columns = rec$dense_features())

  ds <- reticulate::as_iterator(dataset)
  x <- lyr(reticulate::iter_next(ds))

  expect_equal(x$shape$as_list(), c(2, 2 + 2*26))
})

test_that("Recipes are correctly cloned/imutable", {

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b, c) %>%
    step_categorical_column_with_vocabulary_list(a, d)

  rec1 <- rec %>%
    step_indicator_column(a, d)

  rec2 <- rec %>%
    step_indicator_column(a, d)

  rec1$fit()

  expect_length(rec1$features(), 6)
  expect_error(rec2$features())
  expect_error(rec$features())

  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b, c) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  rec_prep <- prep(rec)

  expect_length(rec_prep$features(), 6)
  expect_error(rec$features())
})


test_that("Recipes column types", {
  rec <- recipe(dataset, y ~ a+b+c+d) %>%
    step_numeric_column(b) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  expect_equal(
    rec$feature_types(),
    c("numeric", "nominal", "nominal", "numeric", "numeric", "numeric")
  )
})

test_that("Bake recipe", {

  rec <- recipe(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(b) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  rec_prep <- prep(rec)

  expect_error(bake(rec, dataset))
  expect_s3_class(bake(rec_prep, dataset), "tensorflow.python.data.ops.dataset_ops.DatasetV2")
})

test_that("Juice recipe", {

  rec <- recipe(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(b) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  rec_prep <- prep(rec)

  expect_error(juice(rec))
  expect_s3_class(juice(rec_prep), "tensorflow.python.data.ops.dataset_ops.DatasetV2")
})

test_that("Prep with different dataset", {

  rec <- recipe(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(b) %>%
    step_categorical_column_with_vocabulary_list(a, d) %>%
    step_indicator_column(a, d)

  ds <- df %>%
    tensor_slices_dataset() %>%
    dataset_take(10)

  rec_prep <- prep(rec, ds)

  expect_s3_class(juice(rec_prep), "tensorflow.python.data.ops.dataset_ops.DatasetV2")
})

test_that("Can select with has_type", {

  rec <- recipe(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(has_type("numeric"))

  expect_length(rec$steps, 2)

  rec <- recipe(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(has_type("numeric")) %>%
    step_categorical_column_with_vocabulary_list(has_type("nominal")) %>%
    step_indicator_column(has_type("nominal"))

  expect_length(rec$steps, 6)
  expect_error(rec %>% step_indicator_column(a = has_type("nominal")))

  rec <- recipe(dataset, y ~ a + b + c + d) %>%
    step_numeric_column(all_numeric()) %>%
    step_categorical_column_with_vocabulary_list(has_type("nominal")) %>%
    step_indicator_column(all_nominal())

  expect_length(rec$steps, 6)
})


