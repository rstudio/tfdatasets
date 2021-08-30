
#' Repeats a dataset count times.
#'
#' @param dataset A dataset
#' @param count (Optional.) An integer value representing the number of times
#'   the elements of this dataset should be repeated. The default behavior (if
#'   `count` is `NULL` or `-1`) is for the elements to be repeated indefinitely.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_repeat <- function(dataset, count = NULL) {
  as_tf_dataset(dataset$`repeat`(
    count = as_integer_tensor(count)
  ))
}


#' Randomly shuffles the elements of this dataset.
#'
#' @param dataset A dataset
#'
#' @param buffer_size An integer, representing the number of elements from this
#'   dataset from which the new dataset will sample.
#' @param seed (Optional) An integer, representing the random seed that will be
#'   used to create the distribution.
#' @param reshuffle_each_iteration (Optional) A boolean, which if true indicates
#'   that the dataset should be pseudorandomly reshuffled each time it is iterated
#'   over. (Defaults to `TRUE`). Not used if TF version < 1.15
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_shuffle <- function(dataset, buffer_size, seed = NULL, reshuffle_each_iteration = NULL) {

  if (!is.null(reshuffle_each_iteration) && tensorflow::tf_version() < "1.15")
    warning("reshuffle_each_iteration is only used with TF >= 1.15", call. = FALSE)

  args <- list(
    buffer_size = as_integer_tensor(buffer_size),
    seed = as_integer_tensor(seed)
  )

  if (tensorflow::tf_version() >= "1.15")
    args[["reshuffle_each_iteration"]] <- reshuffle_each_iteration


  as_tf_dataset(do.call(dataset$shuffle, args))
}


#' Shuffles and repeats a dataset returning a new permutation for each epoch.
#'
#' @inheritParams dataset_shuffle
#' @inheritParams dataset_repeat
#'
#' @family dataset methods
#'
#' @export
dataset_shuffle_and_repeat <- function(dataset, buffer_size, count = NULL, seed = NULL) {
  validate_tf_version("1.8", "dataset_shuffle_and_repeat")
  as_tf_dataset(dataset$apply(
    tfd_shuffle_and_repeat(
      as_integer_tensor(buffer_size),
      as_integer_tensor(count),
      as_integer_tensor(seed)
    )
  ))
}

#' Combines consecutive elements of this dataset into batches.
#'
#' @param dataset A dataset
#' @param batch_size An integer, representing the number of consecutive elements
#'   of this dataset to combine in a single batch.
#' @param drop_remainder Ensure that batches have a fixed size by
#'   omitting any final smaller batch if it's present. Note that this is
#'   required for use with the Keras tensor inputs to fit/evaluate/etc.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_batch <-
  function(dataset, batch_size, drop_remainder = FALSE) {
    if (tensorflow::tf_version() > "1.9") {
      as_tf_dataset(
        dataset$batch(batch_size = as_integer_tensor(batch_size),
                      drop_remainder = drop_remainder)
      )
    } else {
      if (drop_remainder) {
        as_tf_dataset(dataset$apply(
          tf$contrib$data$batch_and_drop_remainder(as_integer_tensor(batch_size))
        ))
      } else {
        as_tf_dataset(dataset$batch(batch_size = as_integer_tensor(batch_size)))
      }
    }
}

#' Caches the elements in this dataset.
#'
#'
#' @param dataset A dataset
#' @param filename String with the name of a directory on the filesystem to use
#'   for caching tensors in this Dataset. If a filename is not provided, the
#'   dataset will be cached in memory.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_cache <- function(dataset, filename = NULL) {
  if (is.null(filename))
    filename <- ""
  if (!is.character(filename))
    stop("filename must be a character vector")
  as_tf_dataset(
    dataset$cache(tf$constant(filename, dtype = tf$string))
  )
}


#' Creates a dataset by concatenating given dataset with this dataset.
#'
#' @note Input dataset and dataset to be concatenated should have same nested
#'   structures and output types.
#'
#' @param dataset A dataset
#' @param other Dataset to be concatenated
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_concatenate <- function(dataset, other) {
  as_tf_dataset(dataset$concatenate(other))
}


#' Creates a dataset with at most count elements from this dataset
#'
#' @param dataset A dataset
#' @param count Integer representing the number of elements of this dataset that
#'   should be taken to form the new dataset. If `count` is -1, or if `count` is
#'   greater than the size of this dataset, the new dataset will contain all
#'   elements of this dataset.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_take <- function(dataset, count) {
  as_tf_dataset(dataset$take(count = as_integer_tensor(count)))
}


#' Map a function across a dataset.
#'
#' @param dataset A dataset
#' @param map_func A function mapping a nested structure of tensors (having
#'   shapes and types defined by [output_shapes()] and [output_types()] to
#'   another nested structure of tensors. It also supports `purrr` style
#'   lambda functions powered by [rlang::as_function()].
#' @param num_parallel_calls (Optional) An integer, representing the
#'   number of elements to process in parallel If not specified, elements will
#'   be processed sequentially.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_map <- function(dataset, map_func, num_parallel_calls = NULL) {
  dtype <- if (tensorflow::tf_version() >= "2.3") tf$int64 else tf$int32
  as_tf_dataset(dataset$map(
    map_func = as_py_function(map_func),
    num_parallel_calls = as_integer_tensor(num_parallel_calls, dtype)
  ))
}


#' Fused implementation of dataset_map() and dataset_batch()
#'
#' Maps `map_func`` across batch_size consecutive elements of this dataset and then combines
#' them into a batch. Functionally, it is equivalent to map followed by batch. However, by
#' fusing the two transformations together, the implementation can be more efficient.
#'
#' @inheritParams dataset_map
#' @inheritParams dataset_batch
#' @param num_parallel_batches  (Optional) An integer, representing the number of batches
#'   to create in parallel. On one hand, higher values can help mitigate the effect of
#'   stragglers. On the other hand, higher values can increase contention if CPU is
#'   scarce.
#'
#' @family dataset methods
#'
#' @export
dataset_map_and_batch <- function(dataset,
                                  map_func,
                                  batch_size,
                                  num_parallel_batches = NULL,
                                  drop_remainder = FALSE,
                                  num_parallel_calls = NULL) {
  validate_tf_version("1.8", "dataset_map_and_batch")
  as_tf_dataset(dataset$apply(
    tfd_map_and_batch(
      as_py_function(map_func),
      as.integer(batch_size),
      as_integer_tensor(num_parallel_batches),
      drop_remainder,
      as_integer_tensor(num_parallel_calls)
    )
  ))
}


#' Maps map_func across this dataset and flattens the result.
#'
#' @param dataset A dataset
#'
#' @param map_func A function mapping a nested structure of tensors (having
#'   shapes and types defined by [output_shapes()] and [output_types()] to a
#'    dataset.
#'
#' @return A dataset
#'
#' @export
dataset_flat_map <- function(dataset, map_func) {
  as_tf_dataset(
    dataset$flat_map(map_func)
  )
}

#' Creates a Dataset that prefetches elements from this dataset.
#'
#'
#' @param dataset A dataset
#' @param buffer_size An integer, representing the maximum number elements that
#'   will be buffered when prefetching.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_prefetch <- function(dataset, buffer_size = tf$data$AUTOTUNE) {
  as_tf_dataset(dataset$prefetch(as_integer_tensor(buffer_size)))
}


#' A transformation that prefetches dataset values to the given `device`
#'
#' @param dataset A dataset
#' @param device A string. The name of a device to which elements will be prefetched
#'   (e.g. "/gpu:0").
#' @param buffer_size (Optional.) The number of elements to buffer on device.
#'   Defaults to an automatically chosen value.
#'
#' @return A dataset
#'
#' @note Although the transformation creates a dataset, the transformation must be the
#'   final dataset in the input pipeline.
#'
#' @family dataset methods
#'
#' @export
dataset_prefetch_to_device <- function(dataset, device, buffer_size = NULL) {
  validate_tf_version("1.8", "dataset_prefetch_to_device")
  as_tf_dataset(dataset$apply(
    tfd_prefetch_to_device(
      device = device,
      buffer_size = as_integer_tensor(buffer_size)
    )
  ))
}


#' Filter a dataset by a predicate
#'
#' @param dataset A dataset
#'
#' @param predicate A function mapping a nested structure of tensors (having
#'   shapes and types defined by [output_shapes()] and [output_types()] to a
#'   scalar `tf$bool` tensor.
#'
#' @return A dataset composed of records that matched the predicate.
#'
#' @details Note that the functions used inside the predicate must be
#'   tensor operations (e.g. `tf$not_equal`, `tf$less`, etc.). R
#'   generic methods for relational operators (e.g. `<`, `>`, `<=`,
#'   etc.) and logical operators (e.g. `!`, `&`, `|`, etc.) are
#'   provided so you can use shorthand syntax for most common
#'   comparisions (this is illustrated by the example below).
#'
#' @family dataset methods
#'
#' @examples \dontrun{
#'
#' dataset <- text_line_dataset("mtcars.csv", record_spec = mtcars_spec) %>%
#'   dataset_filter(function(record) {
#'     record$mpg >= 20
#' })
#'
#' dataset <- text_line_dataset("mtcars.csv", record_spec = mtcars_spec) %>%
#'   dataset_filter(function(record) {
#'     record$mpg >= 20 & record$cyl >= 6L
#'   })
#'
#' }
#'
#' @export
dataset_filter <- function(dataset, predicate) {
  as_tf_dataset(dataset$filter(as_py_function(predicate)))
}


#' Creates a dataset that skips count elements from this dataset
#'
#' @param dataset A dataset
#' @param count An integer, representing the number of elements of this dataset
#'   that should be skipped to form the new dataset. If count is greater than
#'   the size of this dataset, the new dataset will contain no elements. If
#'   count is -1, skips the entire dataset.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_skip <- function(dataset, count) {
  as_tf_dataset(dataset$skip(count = as_integer_tensor(count)))
}


#' Maps map_func across this dataset, and interleaves the results
#'
#' @param dataset A dataset
#' @param map_func A function mapping a nested structure of tensors (having
#'   shapes and types defined by [output_shapes()] and [output_types()] to a
#'   dataset.
#' @param cycle_length The number of elements from this dataset that will be
#'   processed concurrently.
#' @param block_length The number of consecutive elements to produce from each
#'   input element before cycling to another input element.
#'
#' @details
#'
#' The `cycle_length` and `block_length` arguments control the order in which
#' elements are produced. `cycle_length` controls the number of input elements
#' that are processed concurrently. In general, this transformation will apply
#' `map_func` to `cycle_length` input elements, open iterators on the returned
#' dataset objects, and cycle through them producing `block_length` consecutive
#' elements from each iterator, and consuming the next input element each time
#' it reaches the end of an iterator.
#'
#' @examples \dontrun{
#'
#' dataset <- tensor_slices_dataset(c(1,2,3,4,5)) %>%
#'  dataset_interleave(cycle_length = 2, block_length = 4, function(x) {
#'    tensors_dataset(x) %>%
#'      dataset_repeat(6)
#'  })
#'
#' # resulting dataset (newlines indicate "block" boundaries):
#' c(1, 1, 1, 1,
#'   2, 2, 2, 2,
#'   1, 1,
#'   2, 2,
#'   3, 3, 3, 3,
#'   4, 4, 4, 4,
#'   3, 3,
#'   4, 4,
#'   5, 5, 5, 5,
#'   5, 5,
#' )
#'
#' }
#'
#' @family dataset methods
#'
#' @export
dataset_interleave <- function(dataset, map_func, cycle_length, block_length = 1) {
  as_tf_dataset(dataset$interleave(
    map_func = as_py_function(map_func),
    cycle_length = as_integer_tensor(cycle_length),
    block_length = as_integer_tensor(block_length)
  ))
}

#' Creates a dataset that includes only 1 / num_shards of this dataset.
#'
#' This dataset operator is very useful when running distributed training, as it
#' allows each worker to read a unique subset.
#'
#' @param dataset A dataset
#' @param num_shards A integer representing the number of shards operating in
#'   parallel.
#' @param index A integer, representing the worker index.
#'
#' @return A dataset
#'
#' @family Dataset methods
#'
#' @export
dataset_shard <- function(dataset, num_shards, index) {
  as_tf_dataset(dataset$shard(
    num_shards = as_integer_tensor(num_shards),
    index = as_integer_tensor(index)
  ))
}


#' Combines consecutive elements of this dataset into padded batches
#'
#' This method combines multiple consecutive elements of this dataset, which
#' might have different shapes, into a single element. The tensors in the
#' resulting element have an additional outer dimension, and are padded to the
#' respective shape in `padded_shapes`.
#'
#' @inheritParams dataset_batch
#'
#' @param dataset A dataset
#' @param batch_size An integer, representing the number of consecutive elements
#'   of this dataset to combine in a single batch.
#' @param padded_shapes A nested structure of tf$TensorShape or integer vector
#'   tensor-like objects representing the shape to which the respective
#'   component of each input element should be padded prior to batching. Any
#'   unknown dimensions (e.g. `tf$Dimension(NULL)` in a `tf$TensorShape` or -1
#'   in a tensor-like object) will be padded to the maximum size of that
#'   dimension in each batch.
#' @param padding_values (Optional) A nested structure of scalar-shaped
#'   tf$Tensor, representing the padding values to use for the respective
#'   components. Defaults are 0 for numeric types and the empty string for
#'   string types.
#'
#' @return A dataset
#'
#' @family dataset methods
#'
#' @export
dataset_padded_batch <- function(dataset, batch_size, padded_shapes, padding_values = NULL,
                                 drop_remainder = FALSE) {
  if (drop_remainder) {
    as_tf_dataset(dataset$apply(
      tf$contrib$data$padded_batch_and_drop_remainder(
        as_integer_tensor(batch_size),
        as_tensor_shapes(padded_shapes),
        as_integer_tensor(padding_values)
      )
    ))
  } else {
    as_tf_dataset(dataset$padded_batch(
      batch_size = as_integer_tensor(batch_size),
      padded_shapes = as_tensor_shapes(padded_shapes),
      padding_values = padding_values
    ))
  }
}



#' Prepare a dataset for analysis
#'
#' Transform a dataset with named columns into a list with features (`x`) and
#' response (`y`) elements.
#'
#' @inheritParams dataset_decode_delim
#'
#' @param dataset A dataset
#'
#' @param x Features to include. When `named_features` is `FALSE` all features
#'   will be stacked into a single tensor so must have an identical data type.
#'
#' @param y (Optional). Response variable.
#'
#' @param named `TRUE` to name the dataset elements "x" and "y", `FALSE` to
#'   not name the dataset elements.
#'
#' @param named_features `TRUE` to yield features as a named list; `FALSE` to
#'   stack features into a single array. Note that in the case of `FALSE` (the
#'   default) all features will be stacked into a single 2D tensor so need to
#'   have the same underlying data type.
#'
#' @param batch_size (Optional). Batch size if you would like to fuse the
#'   `dataset_prepare()` operation together with a `dataset_batch()` (fusing
#'   generally improves overall training performance).
#'
#' @inheritParams dataset_map_and_batch
#'
#' @return A dataset. The dataset will have a structure of either:
#'
#'   - When `named_features` is `TRUE`: `list(x = list(feature_name = feature_values, ...), y = response_values)`
#'
#'   - When `named_features` is `FALSE`: `list(x = features_array, y = response_values)`,
#'     where `features_array` is a Rank 2 array of `(batch_size, num_features)`.
#'
#' Note that the `y` element will be omitted when `y` is `NULL`.
#'
#' @seealso [input_fn()][input_fn.tf_dataset()] for use with \pkg{tfestimators}.
#'
#' @export
dataset_prepare <- function(dataset, x, y = NULL, named = TRUE, named_features = FALSE,
                            parallel_records = NULL,
                            batch_size = NULL,
                            num_parallel_batches = NULL,
                            drop_remainder = FALSE) {

  # validate dataset
  if (!is_dataset(dataset))
    stop("Provided dataset is not a TensorFlow Dataset")

  # default to null response_col
  response_col <- NULL

  # get features
  col_names <- column_names(dataset)
  eq_features <- rlang::enquo(x)

  # attempt use of tidyselect. if there is an error it could be because 'x'
  # is a formula. in that case attempt to parse the formula
  feature_col_names <- tryCatch({
    tidyselect::vars_select(col_names, !! eq_features)
  },
  error = function(e) {


    x <- get_expr(eq_features)
    if (is_formula(x)) {

      data <- lapply(column_names(dataset), function(x) "")
      names(data) <- column_names(dataset)
      data <- as.data.frame(data)

      parsed <- parse_formula(x, data)
      if (!is.null(parsed$response))
        response_col <<- match(parsed$response, col_names)
      parsed$features
    } else {
      stop(e$message, call. = FALSE)
    }
  })

  # get column indexes
  feature_cols <- match(feature_col_names, col_names)


  # get response if specified
  if (!missing(y) && is.null(response_col)) {
    eq_response <- rlang::enquo(y)
    response_name <- tidyselect::vars_select(col_names, !! eq_response)
    if (length(response_name) > 0) {
      if (length(response_name) != 1)
        stop("Invalid response column: ", paste(response_name))
      response_col <- match(response_name, col_names)
    }
  }

  # mapping function
  map_func <- function(record) {

    # `make_csv_dataset` returns an ordered dict instead of a `dict`
    # which in turn doesn't get automatically converted by reticulate.
    if (inherits(record, "python.builtin.dict"))
      record <- reticulate::py_to_r(record)

    # select features
    record_features <- record[feature_cols]

    # apply names to features if named
    if (named_features) {

      names(record_features) <- feature_col_names

      # otherwise stack features into a single tensor
    } else {
      record_features <- unname(record_features)
      # determine the axis based on the shape of the tensor
      # (unbatched tensors will be scalar with no shape,
      #  so will stack on axis 0)
      shape <- record_features[[1]]$get_shape()$as_list()
      axis <- length(shape)
      record_features <- tf$stack(record_features, axis = axis)
    }

    # massage the record into the approriate structure
    if (!is.null(response_col)) {
      record <- list(record_features, record[[response_col]])
      if (named)
        names(record) <- c("x", "y")
    }
    else {
      record <- list(record_features)
      if (named)
        names(record) <- c("x")
    }

    # return the record
    record
  }


  # call appropriate mapping function
  if (is.null(batch_size)) {
    dataset <- dataset %>%
      dataset_map(map_func = map_func,
                  num_parallel_calls = parallel_records)
  } else {
    dataset <- dataset %>%
      dataset_map_and_batch(map_func = map_func,
                            batch_size = batch_size,
                            num_parallel_batches = num_parallel_batches,
                            drop_remainder = drop_remainder,
                            num_parallel_calls = parallel_records)
  }

  # return dataset
  as_tf_dataset(dataset)
}


#' Add the tf_dataset class to a dataset
#'
#' Calling this function on a dataset adds the "tf_dataset" class to the dataset
#' object. All datasets returned by functions in the \pkg{tfdatasets} package
#' call this function on the dataset before returning it.
#'
#' @param dataset A dataset
#'
#' @return A dataset with class "tf_dataset"
#'
#' @keywords internal
#'
#' @export
as_tf_dataset <- function(dataset) {

  # validate dataset
  if (!is_dataset(dataset))
    stop("Provided dataset is not a TensorFlow Dataset")

  # add class if needed
  if (!inherits(dataset, "tf_dataset"))
  class(dataset) <- c("tf_dataset", class(dataset))

  # return
  dataset
}


#' Combines input elements into a dataset of windows.
#'
#' @param dataset A dataset
#' @param size representing the number of elements of the input dataset to
#'    combine into a window.
#' @param shift epresenting the forward shift of the sliding window in each
#'    iteration. Defaults to `size`.
#' @param stride representing the stride of the input elements in the sliding
#'    window.
#' @param drop_remainder representing whether a window should be dropped in
#'    case its size is smaller `than window_size`.
#'
#' @family dataset methods
#'
#' @export
dataset_window <- function(dataset, size, shift = NULL, stride = 1,
                           drop_remainder = FALSE) {
  as_tf_dataset(
    dataset$window(
      size = as_integer_tensor(size),
      shift = as_integer_tensor(shift),
      stride = as_integer_tensor(stride),
      drop_remainder = drop_remainder
    )
  )
}

#' Collects a dataset
#'
#' Iterates throught the dataset collecting every element into a list.
#' It's useful for looking at the full result of the dataset.
#' Note: You may run out of memory if your dataset is too big.
#'
#' @param dataset A dataset
#' @param iter_max Maximum number of iterations. `Inf` until the end of the
#'  dataset
#'
#' @family dataset methods
#'
#' @export
dataset_collect <- function(dataset, iter_max = Inf) {

  if (tensorflow::tf_version() < "2.0")
    stop("dataset_collect requires TF 2.0", call.=FALSE)

  it <- reticulate::as_iterator(dataset)

  out <- list()
  i <- 0

  while(!is.null(x <- reticulate::iter_next(it))) {
    i <- i + 1
    out[[i]] <- x
    if (i >= iter_max) break
  }

  out
}

#' Reduces the input dataset to a single element.
#'
#' The transformation calls reduce_func successively on every element of the input dataset
#' until the dataset is exhausted, aggregating information in its internal state.
#' The initial_state argument is used for the initial state and the final state is returned as the result.
#'
#' @param dataset A dataset
#' @param initial_state An element representing the initial state of the transformation.
#' @param reduce_func A function that maps `(old_state, input_element)` to new_state.
#' It must take two arguments and return a new element.
#' The structure of new_state must match the structure of initial_state.
#'
#' @return A dataset element.
#'
#' @family dataset methods
#'
#' @export
dataset_reduce <- function(dataset, initial_state, reduce_func) {
  dataset$reduce(initial_state, reduce_func)
}


#' Get or Set Dataset Options
#'
#' @param dataset a tensorflow dataset
#' @param ... Valid values include:
#'
#'   +  A set of named arguments setting options. Names of nested attributes can
#'   be separated with a `"."` (see examples). The set of named arguments can be
#'   supplied individually to `...`, or as a single named list.
#'
#'   + a `tf$data$Options()` instance.
#'
#'
#' @return If values are supplied to `...`, returns a `tf.data.Dataset` with the
#'   given options set/updated. Otherwise, returns the currently set options for
#'   the dataset.
#'
#' @details The options are "global" in the sense they apply to the entire
#'   dataset. If options are set multiple times, they are merged as long as
#'   different options do not use different non-default values.
#'
#'
#' @export
#' @examples
#' \dontrun{
#' # pass options directly:
#' range_dataset(0, 10) %>%
#'   dataset_options(
#'     experimental_deterministic = FALSE,
#'     threading.private_threadpool_size = 10
#'   )
#'
#' # pass options as a named list:
#' opts <- list(
#'   experimental_deterministic = FALSE,
#'   threading.private_threadpool_size = 10
#' )
#' range_dataset(0, 10) %>%
#'   dataset_options(opts)
#'
#' # pass a tf.data.Options() instance
#' opts <- tf$data$Options()
#' opts$experimental_deterministic <- FALSE
#' opts$threading$private_threadpool_size <- 10L
#' range_dataset(0, 10) %>%
#'   dataset_options(opts)
#'
#' # get currently set options
#' range_dataset(0, 10) %>% dataset_options()
#' }
dataset_options <- function(dataset, ...) {
  user_opts <- list(...)

  if(!length(user_opts))
    return(dataset$options())

  options <- tf$data$Options()

  # accept a packed list of arguments, don't required do.call for programming
  if(is.null(names(user_opts)) &&
     length(user_opts) == 1 &&
     is.list(user_opts[[1]]))
    user_opts <- user_opts[[1]]

  for (i in seq_along(user_opts)) {
    name <- names(user_opts)[i]
    val <- user_opts[[i]]

    if (inherits(val, c("tensorflow.python.data.ops.dataset_ops.Options",
                        "tensorflow.python.data.ops.options.Options"))) {
      options <- options$merge(val)
      next
    }

    # special convenience hooks for some known options, with a no-op fallback
    transform <- switch(name,
      "threading.private_threadpool_size" = as.integer,
      "threading.max_intra_op_parallelism" = as.integer,
      "experimental_distribute.num_devices" = as.integer,
      identity
    )

    val <- transform(val)

    # change names like "foo.bar.baz" to an R expression like
    # `options$foo$bar$baz`, but with some semblance of safety by avoiding
    # parse(), using as.symbol() on user supplied names, and constructing the
    # call we want directly. We do this to avoid hand-coding a recursive impl
    # using py_set_attr(), and let the R's `$<-` method do the recursion.
    target <- Reduce(
      function(x, y) substitute(x$y, list(x = x, y = as.symbol(y))),
      strsplit(name, ".", fixed = TRUE)[[1]],
      init = quote(options))

    expr <- substitute(target <- val, list(target = target))
    eval(expr)
  }

  as_tf_dataset(dataset$with_options(options))
}


#' Get Dataset length
#'
#' Returns the length of the dataset.
#'
#' @param x a `tf.data.Dataset` object.
#'
#' @return Either `Inf` if the dataset is infinite, `NA` if the dataset length
#'   is unknown, or an R numeric if it is known.
#' @export
#' @importFrom tensorflow tf_version
#' @examples
#' \dontrun{
#' range_dataset(0, 42) %>% length()
#' # 42
#'
#' range_dataset(0, 42) %>% dataset_repeat() %>% length()
#' # Inf
#'
#' range_dataset(0, 42) %>% dataset_repeat() %>%
#'   dataset_filter(function(x) TRUE) %>% length()
#' # NA
#' }
length.tf_dataset <- function(x) {
  if (tf_version() >= "2.3") {
    l <- x$cardinality()$numpy()
    car_inf <- tf$data$INFINITE_CARDINALITY
    car_unk <- tf$data$UNKNOWN_CARDINALITY
  } else {
    l <- tf$data$experimental$cardinality(x)$numpy()
    car_inf <- tf$data$experimental$INFINITE_CARDINALITY
    car_unk <- tf$data$experimental$UNKNOWN_CARDINALITY
  }

  if (l == car_inf)
    Inf
  else if (l == car_unk)
    NA
  else
    l
}

#' @export
#' @rdname length.tf_dataset
length.tensorflow.python.data.ops.dataset_ops.DatasetV2 <- length.tf_dataset


#' Enumerates the elements of this dataset
#'
#' @details It is similar to python's `enumerate`, this transforms a sequence of
#' elements into a sequence of `list(index, element)`, where index is an integer
#' that indicates the position of the element in the sequence.
#'
#' @param dataset A tensorflow dataset
#' @param start An integer (coerced to a `tf$int64` scalar `tf.Tensor`),
#'   representing the start value for enumeration.
#'
#' @export
#' @examples
#' \dontrun{
#' dataset <- tensor_slices_dataset(100:103) %>%
#'   dataset_enumerate()
#'
#' iterator <- reticulate::as_iterator(dataset)
#' reticulate::iter_next(iterator) # list(0, 100)
#' reticulate::iter_next(iterator) # list(1, 101)
#' reticulate::iter_next(iterator) # list(2, 102)
#' reticulate::iter_next(iterator) # list(3, 103)
#' reticulate::iter_next(iterator) # NULL (iterator exhausted)
#' reticulate::iter_next(iterator) # NULL (iterator exhausted)
#' }
dataset_enumerate <- function(dataset, start=0L) {
  as_tf_dataset(dataset$enumerate(as_integer_tensor(start)))
}


#' Creates a `Dataset` of pseudorandom values
#'
#' @details
#' The dataset generates a sequence of uniformly distributed integer values (dtype int64).
#'
#' @param seed (Optional) If specified, the dataset produces a deterministic
#' sequence of values.
#'
#' @export
random_integer_dataset <- function(seed = NULL) {
  if (tf_version() >= "2.6")
    as_tf_dataset(tf$data$Dataset$random(as_integer_tensor(seed)))
  else
    as_tf_dataset(tf$data$experimental$RandomDataset(as_integer_tensor(seed)))
}


#' A transformation that scans a function across an input dataset
#'
#' @details
#' This transformation is a stateful relative of `dataset_map()`.
#' In addition to mapping `scan_func` across the elements of the input dataset,
#' `scan()` accumulates one or more state tensors, whose initial values are
#' `initial_state`.
#'
#' @param dataset A tensorflow dataset
#'
#' @param initial_state A nested structure of tensors, representing the initial
#' state of the accumulator.
#'
#' @param scan_func A function that maps `(old_state, input_element)` to
#' `(new_state, output_element)`. It must take two arguments and return a
#' pair of nested structures of tensors. The `new_state` must match the
#' structure of `initial_state`.
#'
#' @export
#' @examples
#' \dontrun{
#' initial_state <- as_tensor(0, dtype="int64")
#' scan_func <- function(state, i) list(state + i, state + i)
#' dataset <- range_dataset(0, 10) %>%
#'   dataset_scan(initial_state, scan_func)
#'
#' reticulate::iterate(dataset, as.array) %>%
#'   unlist()
#' # 0  1  3  6 10 15 21 28 36 45
#' }
dataset_scan <- function(dataset, initial_state, scan_func) {
  if(tf_version() >= "2.6")
    as_tf_dataset(dataset$scan(initial_state, as_py_function(scan_func)))
  else {
    dataset$apply(
      tf$data$experimental$scan(initial_state, as_py_function(scan_func))
    )
  }
}


#' Persist the output of a dataset
#'
#' @details
#' The snapshot API allows users to transparently persist the output of their
#' preprocessing pipeline to disk, and materialize the pre-processed data on a
#' different training run.
#'
#' This API enables repeated preprocessing steps to be consolidated, and allows
#' re-use of already processed data, trading off disk storage and network
#' bandwidth for freeing up more valuable CPU resources and accelerator compute
#' time.
#'
#' https://github.com/tensorflow/community/blob/master/rfcs/20200107-tf-data-snapshot.md
#' has detailed design documentation of this feature.
#'
#' Users can specify various options to control the behavior of snapshot,
#' including how snapshots are read from and written to by passing in
#' user-defined functions to the `reader_func` and `shard_func` parameters.
#'
#' `shard_func` is a user specified function that maps input elements to
#' snapshot shards.
#'
# If `shard_func` is not supplied, the equivalent action is performed:
#' ```R
#' NUM_SHARDS <- parallel::detectCores()
#' dataset %>%
#'   dataset_enumerate() %>%
#'   dataset_snapshot(
#'     "/path/to/snapshot/dir",
#'     shard_func = function(index, ds_elem) x %% NUM_SHARDS) %>%
#'   dataset_map(function(index, ds_elem) ds_elem)
#' ```
#'
#' `reader_func` is a user specified function that accepts a single argument:
#' a Dataset of Datasets, each representing a "split" of elements of the
#' original dataset. The cardinality of the input dataset matches the
#' number of the shards specified in the `shard_func`. The function
#' should return a Dataset of elements of the original dataset.
#'
#' Users may want specify this function to control how snapshot files should be
#' read from disk, including the amount of shuffling and parallelism.
#'
#' Here is an example of a standard reader function a user can define. This
#' function enables both dataset shuffling and parallel reading of datasets:
#'
#' ````R
#' user_reader_func <- function(datasets) {
#'   num_cores <- parallel::detectCores()
#'   datasets %>%
#'     dataset_shuffle(num_cores) %>%
#'     dataset_interleave(function(x) x, num_parallel_calls=AUTOTUNE)
#' }
#'
#' dataset <- dataset %>%
#'   dataset_snapshot("/path/to/snapshot/dir",
#'                    reader_func = user_reader_func)
#' ````
#'
#' By default, snapshot parallelizes reads by the number of cores available on
#' the system, but will not attempt to shuffle the data.
#'
#' @param dataset A tensorflow dataset
#'
#' @param path Required. A directory to use for storing/loading the snapshot to/from.
#'
#' @param compression Optional. The type of compression to apply to the snapshot
#'   written to disk. Supported options are `"GZIP"`, `"SNAPPY"`, `"AUTO"` or
#'   `NULL` (values of `""`, `NA`, and `"None"` are synonymous with `NULL`)
#'   Defaults to `AUTO`, which attempts to pick an appropriate compression
#'   algorithm for the dataset.
#'
#' @param reader_func Optional. A function to control how to read data from
#' snapshot shards.
#'
#' @param shard_func Optional. A function to control how to shard data when writing
#' a snapshot.
#'
#' @export
dataset_snapshot <- function(dataset, path, compression=c("AUTO", "GZIP", "SNAPPY", "None"),
                             reader_func=NULL, shard_func=NULL) {
  if(identical(compression, ""))
    compression <- NULL
  else if(!is.null(compression)) {
    compression <- match.arg(compression)
    if(compression == "None")
      compression <- NULL
  }

  if (!is.null(reader_func))
    reader_func <- as_py_function(reader_func)
  if (!is.null(shard_func))
    shard_func <- as_py_function(shard_func)

  args <- list(path, compression=compression,
                   reader_func = reader_func,
                   shard_func = shard_func)

  if(tf_version() >= "2.6")
    do.call(dataset$snapshot, args)
  else
    dataset$apply(do.call(tf$data$experimental$snapshot, args))
}
