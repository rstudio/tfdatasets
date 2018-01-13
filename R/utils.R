


as_integer_tensor <- function(x, dtype = tf$int64) {
  # recurse over lists
  if (is.list(x) || (is.numeric(x) && length(x) > 1))
    lapply(x, function(elem) as_integer_tensor(elem, dtype))
  else if (is.null(x))
    x
  else if (is_tensor(x))
    tf$cast(x, dtype = dtype)
  else
    tf$constant(as.integer(x), dtype = dtype)
}

as_tensor_shapes <- function(x) {
  if (is.list(x))
    tuple(lapply(x, as_tensor_shapes))
  else if (is_tensor(x))
    tf$cast(x, dtype = tf$int64)
  else if (inherits("x", "python.builtin.object"))
    x
  else if (is.null(x))
    tf$constant(-1L, dtype = tf$int64)
  else
    tf$constant(as.integer(x), dtype = tf$int64)
}


with_session <- function(f, session = NULL) {
  if (is.null(session))
    session <- tf$get_default_session()
  if (is.null(session)) {
    session <- tf$Session()
    on.exit(session$close(), add = TRUE)
  }
  f(session)
}


validate_tf_version <- function() {
  tf_ver <- tensorflow::tf_version()
  required_ver <- "1.4"
  if (tf_ver < required_ver) {
    stop(
      "tfdatasets requires version ", required_ver, " ",
      "of TensorFlow (you are currently running version ", tf_ver, ").",
      call. = FALSE
    )
  }
}

column_names <- function(dataset) {
  if (!is.list(dataset$output_shapes) || is.null(names(dataset$output_shapes)))
    stop("Unable to resolve features for dataset that does not have named outputs", call. = FALSE)
  names(dataset$output_shapes)
}

is_dataset <- function(x) {
  inherits(x, "tensorflow.python.data.ops.dataset_ops.Dataset")
}

is_tensor <- function(x) {
  inherits(x, "tensorflow.python.framework.ops.Tensor")
}



