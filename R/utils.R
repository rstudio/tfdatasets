

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
  else if (inherits(x, "python.builtin.object"))
    x
  else if (is.null(x))
    tf$constant(-1L, dtype = tf$int64)
  else
    tf$constant(as.integer(x), dtype = tf$int64)
}

with_session <- function(f, session = NULL) {
  if (is.null(session)) {
    if (tensorflow::tf_version() >= "1.14")
      session <- tensorflow::tf$compat$v1$get_default_session()
  } else {
    session <- tf$get_default_session()
  }
  if (is.null(session)) {
    if (tensorflow::tf_version() >= "1.14")
      session <- tf$compat$v1$Session()
    else
      session <- tf$Session()
    on.exit(session$close(), add = TRUE)
  }
  f(session)
}


validate_tf_version <- function(required_ver = "1.4", feature_name = "tfdatasets") {
  tf_ver <- tensorflow::tf_version()
  if (is.null(tf_ver)) {
    stop("You need to install TensorFlow to use tfdatasets ",
         "-- install with tensorflow::install_tensorflow()",
         call. = FALSE)
  } else if (tf_ver < required_ver) {
    stop(
      feature_name, " requires version ", required_ver, " ",
      "of TensorFlow (you are currently running version ", tf_ver, ").",
      call. = FALSE
    )
  }
}

column_names <- function(dataset) {

  if (tensorflow::tf_version() >= "2.0") {
    x <- next_batch(dataset)
  } else {
    x <- dataset$output_shapes
  }

  if (!is.list(x) || is.null(names(x)))
    stop("Unable to resolve features for dataset that does not have named outputs", call. = FALSE)

  names(x)
}

is_dataset <- function(x) {
  inherits(x, "tensorflow.python.data.ops.dataset_ops.Dataset") ||
  inherits(x, "tensorflow.python.data.ops.dataset_ops.DatasetV2")
}

is_tensor <- function(x) {
  inherits(x, "tensorflow.python.framework.ops.Tensor")
}

is_eager_tensor <- function(x) {
  inherits(x, "python.builtin.EagerTensor") ||
  inherits(x, "tensorflow.python.framework.ops.EagerTensor")
}

as_py_function <- function(x) {
  if (inherits(x, "python.builtin.function")) {
    x
  } else {
    rlang::as_function(x)
  }
}


