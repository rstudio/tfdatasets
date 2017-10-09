


as_integer_tensor <- function(x, dtype = tf$int64) {
  if (is.null(x))
    x
  else if (inherits(x, "tensorflow.python.framework.ops.Tensor"))
    tf$cast(x, dtype = dtype)
  else
    tf$constant(as.integer(x), dtype = dtype)
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

  # resolve filename wildcards then iterate over the results
  filenames <- tf$contrib$data$Dataset$list_files(filenames)
  iter <- one_shot_iterator(filenames)
  with_session(function(sess) {
    all_filenames <- character()
    while(!is.null(filename <- iterator_next(iter, sess)))
      all_filenames <- c(all_filenames, filename)
    all_filenames
  })
}


