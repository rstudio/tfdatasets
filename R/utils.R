

as_integer_tensor <- function(x, dtype = tf$int64) {

  # recurse over lists
  if (is.list(x) || (is.numeric(x) && length(x) > 1))
    lapply(x, function(elem) as_integer_tensor(elem, dtype))
  else if (is.null(x))
    x
  else if (is_tensor(x))
    tf$cast(x, dtype = dtype)
  else
    as.integer(x)
    # https://github.com/tensorflow/tensorflow/issues/71937
    # tf$constant(as.integer(x), dtype = dtype)
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


as_integer_list <- function(x) as.list(as.integer(x))


# assert_all_dots_named(), capture_args(), require_tf_version()
# copy-pasted from keras circa tf_version() 2.7
require_tf_version <- function(ver, msg = "this function.") {
  if (tf_version() < ver)
    stop("Tensorflow version >=", ver, " required to use ", msg)
}

assert_all_dots_named <- function(envir = parent.frame(), cl) {

  x <- eval(quote(list(...)), envir)
  if(!length(x))
    return()

  x <- names(x)
  if(is.character(x) && !anyNA(x) && all(x != ""))
    return()

  stop("All arguments provided to `...` must be named.\n",
       "Call with unnamed arguments in dots:\n  ",
       paste(deparse(cl, 500L), collapse = "\n"))
}

# TODO: should there be some default modifiers in capture_args() for standard layer args
# like, input_shape, batch_input_shape, etc.

capture_args <- function(cl, modifiers = NULL, ignore = NULL,
                         envir = parent.frame(), fn = sys.function(-1)) {

  ## bug: match.call() resolves incorrectly if dots are from not the default sys.parent()
  ## e.g, this fails if dots originate from the callers caller:
  #    cl <- eval(quote(match.call()), parent.frame())
  ## workaround: caller must call match.call() from the correct frame

  ## note: capture_args() must always be called at the top level of the intended function body.
  ## sys.function(-1) resolves to the incorrect function if the  capture_args()
  ## call is itself a promise in another call. E.g.,:
  ##   do.call(foo, capture_args(match.call())) fails because fn resolves to do.call()

  fn_arg_nms <- names(formals(fn))
  known_args <- intersect(names(cl), fn_arg_nms)
  known_args <- setdiff(known_args, ignore)
  names(known_args) <- known_args
  cl2 <- c(quote(list), lapply(known_args, as.symbol))

  if("..." %in% fn_arg_nms && !"..." %in% ignore) {
    assert_all_dots_named(envir, cl)
    # this might reorder args by assuming ... are last, but it doesn't matter
    # since everything is supplied as a keyword arg to the Python side anyway
    cl2 <- c(cl2, quote(...))
  }

  args <- eval(as.call(cl2), envir)

  for(nm in intersect(names(args), ignore))
    args[[nm]] <- NULL

  nms_to_modify <- intersect(names(args), names(modifiers))
  for (nm in nms_to_modify)
    args[nm] <- list(modifiers[[nm]](args[[nm]]))
   # list() so if modifier returns NULL, don't remove the arg

  args
}
