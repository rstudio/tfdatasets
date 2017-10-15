library(testthat)
library(tensorflow)
library(tfdatasets)

# run tests in default tf session
sess <- tf$Session()
with(sess$as_default(), {
  test_check("tfdatasets")
})






