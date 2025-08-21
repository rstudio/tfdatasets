library(testthat)
library(tensorflow)
library(tfdatasets)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  reticulate::py_require(
    if (Sys.info()[["sysname"]] == "Linux") "tensorflow-cpu" else "tensorflow"
  )
  test_check("tfdatasets")
}
