
library(tensorflow)


skip_if_no_tensorflow <- function(required_version = NULL) {
  if (!reticulate::py_module_available("tensorflow"))
    skip("TensorFlow not available for testing")
  else if (!is.null(required_version)) {
    if (tensorflow::tf_version() < required_version)
      skip(sprintf("Required version of TensorFlow (%s) not available for testing",
                   required_version))
  }
}

skip_if_eager <- function(message) {
  if (tf$executing_eagerly())
    skip(message)
}

skip_if_v2 <- function(message) {
  if (tensorflow::tf_version() >= "2.0")
    skip(message)
}

test_succeeds <- function(desc, expr, required_version = NULL) {
  test_that(desc, {
    skip_if_no_tensorflow(required_version)
    expect_error(force(expr), NA)
  })
}

csv_dataset <- function(file, ...) {
  csv_spec <- csv_record_spec(file, ...)
  text_line_dataset(file, record_spec = csv_spec)
}

mtcars_dataset <- function() {
  csv_dataset("data/mtcars.csv") %>%
    dataset_shuffle(50) %>%
    dataset_batch(10)
}

mtcars_dataset_nobatch <- function() {
  csv_dataset("data/mtcars.csv") %>%
    dataset_shuffle(50)
}


