library(testthat)
library(tensorflow)
library(tfdatasets)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  test_check("tfdatasets")
}
