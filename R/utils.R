

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



#' @importFrom rlang list2
capture_args <- function(modifiers = NULL, ignore = NULL, force = NULL,
                         enforce_all_dots_named = TRUE) {
  call <- sys.call(-1L)
  envir <- parent.frame(1L)
  fn <- sys.function(-1L)
  # if("capture_args" %in% all.names(call, unique = TRUE))
  #   stop("incorrect usage of capture_args(), must be evaluated as ",
  #        "a standard expression, not as not a promise (i.e., not as part ",
  #         "of a call of another function")

  # match.call() automatically omits missing() args in the returned call. These
  # user calls all standardize to the same thing:
  # - layer_dense(, 10)
  # - layer_dense(object = , 10)
  # - layer_dense(object = , 10, )
  # - layer_dense(, 10, )
  # all standardize to:
  # - layer_dense(units = 10)
  call <- match.call(fn, call, expand.dots = TRUE, envir = parent.frame(2))

  # message("call: ", deparse1(call))

  fn_arg_nms <- names(formals(fn))
  known_args <- intersect(names(call), fn_arg_nms)
  if (length(ignore) && !is.character(ignore)) {
    # e.g., ignore = c("object", \(nms) startsWith(nms, "."))
    ignore <- as.character(unlist(lapply(ignore, function(ig) {
      if (is.character(ig)) return(ig)
      stopifnot(is.function(ig))
      ig <- ig(known_args) # ignore fn can return either lgl or int for [
      if (!is.character(ig))
        ig <- known_args[ig]
      ig
    }), use.names = FALSE))
  }
  known_args <- setdiff(known_args, ignore)
  known_args <- union(known_args, force)
  names(known_args) <- known_args

  if ("..." %in% fn_arg_nms && !"..." %in% ignore) {
    if (enforce_all_dots_named)
      assert_all_dots_named(envir, call)
    # match.call already drops missing args that match to known args, but it
    # doesn't protect from missing args that matched into ...
    # use list2() to allow dropping a trailing missing arg in ... also
    dots <- quote(...)
    list_sym <- quote(list2)
  } else {
    dots <- NULL
    list_sym <- quote(list)
  }

  # this might reorder args by assuming ... are last, but it doesn't matter
  # since everything is supplied as a keyword arg to the Python side anyway
  call <- as.call(c(list_sym, lapply(known_args, as.symbol), dots))
  args <- eval(call, envir)

  # filter out ignore again, in case any were in ...
  # we could probably enhance the `call` constructed above to use, e.g.,
  # ..1, ..2, ..4, to skip ignores, and avoid forcing them.
  if (length(ignores_in_dots <- intersect(names(call), ignore)))
    args[ignores_in_dots] <- NULL

  # apply modifier functions. e.g., as_nullable_integer()
  if (length(names_to_modify <-
             intersect(names(args), names(modifiers))))
    args[names_to_modify] <-
    map2(modifiers[names_to_modify], args[names_to_modify],
         function(modifier, arg) modifier(arg))

  args
}


map2 <- function(.x, .y, .f, ...) {
  out <- .mapply(.f, list(.x, .y), list(...))
  if (length(.x) == length(out))
    names(out) <- names(.x)
  out
}
