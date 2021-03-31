

#' @importFrom reticulate py_last_error tuple py_str py_has_attr
#' @import tidyselect
#' @import rlang
NULL

.onLoad <- function(libname, pkgname) {
  vctrs::s3_register("tfestimators::input_fn", "tf_dataset")
}


