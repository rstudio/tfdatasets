
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

skip_if_not_eager <- function() {
  if (!tf$executing_eagerly())
    skip("Eager execution is required. ")
}

skip_if_v2 <- function(message) {
  if (tensorflow::tf_version() >= "2.0")
    skip(message)
}

skip_tfestimators <- function() {
  skip("tfestimators deprecated")
}

py_capture_output <- reticulate::py_capture_output

test_succeeds <- function(desc, expr, required_version = NULL) {
  # IPython <- reticulate::import("IPython")
  # py_capture_output <- IPython$utils$capture$capture_output
  invisible(
    capture.output({
      py_capture_output({
        test_that(desc, {
          skip_if_no_tensorflow(required_version)
          expect_error(force(expr), NA)
        })
      })
    })
  )
}

csv_dataset <- function(file, ...) {
  csv_spec <- csv_record_spec(file, ...)
  text_line_dataset(file, record_spec = csv_spec)
}

mtcars_dataset <- function() {
  testing_data_filepath("mtcars.csv") %>%
    csv_dataset() %>%
    dataset_shuffle(50) %>%
    dataset_batch(10)
}

mtcars_dataset_nobatch <- function() {
  testing_data_filepath("mtcars.csv") %>%
    csv_dataset() %>%
    dataset_shuffle(50)
}


testing_data_filepath <- function(x = NULL) {
  # basename(getwd()) is either "tfdatasets" or "tests" if in R CMD check
  d <- if(file.exists("data/iris.csv"))
     "data" else "tests/testthat/data"
  if (is.null(x))
    d
  else
    file.path(d, x)
}
