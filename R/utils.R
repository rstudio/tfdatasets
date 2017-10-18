


as_integer_tensor <- function(x, dtype = tf$int64) {
  # recurse over lists
  if (is.list(x) || (is.numeric(x) && length(x) > 1))
    lapply(x, function(elem) as_integer_tensor(elem, dtype))
  else if (is.null(x))
    x
  else if (inherits(x, "tensorflow.python.framework.ops.Tensor"))
    tf$cast(x, dtype = dtype)
  else
    tf$constant(as.integer(x), dtype = dtype)
}

as_tensor_shape <- function(x) {
  # reflect TensorShape back
  if (inherits(x, "tensorflow.python.framework.tensor_shape.TensorShape"))
    x
  else
    as_integer_tensor(x)
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

auto_compression_type <- function(filenames) {
  has_ext <- function(ext) {
    any(identical(tolower(tools::file_ext(filenames)), ext))
  }
  if (has_ext("gz"))
    "GZIP"
  else if (has_ext("zlib"))
    "ZLIB"
  else
    ""
}

resolve_filenames <- function(filenames) {

  # vectorize
  if (length(filenames) > 1) {
    all_filenames <- character()
    for (filename in filenames)
      all_filenames <- c(all_filenames, resolve_filenames(filename))
    return(all_filenames)
  }

  # list files and return all results
  filenames <- file_list_dataset(filenames) %>%
    dataset_take(-1)
  batch <- next_batch(filenames)
  with_session(function(sess) {
    sess$run(batch)
  })
}


validate_tf_version <- function() {
  tf_ver <- tensorflow::tf_version()
  required_ver <- "1.4"
  if (tf_ver < required_ver) {
      stop(
        "tfdatasets requires version ", required_ver, " ",
        "of TensorFlow (you are currently running version ", tf_ver, ").\n",
        "Please update your TensorFlow to nightly builds following the instruction here: \n",
        "https://tensorflow.rstudio.com/tools/installation.html#alternate-versions",
        call. = FALSE
      )
  }
}

column_names <- function(dataset) {
  if (!is.list(dataset$output_shapes) || is.null(names(dataset$output_shapes)))
    stop("Unable to resolve features for dataset that does not have named outputs", call. = FALSE)
  names(dataset$output_shapes)
}


