name_step <- function(step) {
  if (!is.null(step$name)) {
    nm <- step$name
    step <- list(step)
    names(step) <- nm
  }

  step
}

is_dense_column <- function(feature) {
  inherits(feature, "tensorflow.python.feature_column.feature_column._DenseColumn")
}

dtype_chr <- function(x) {
  if (x == tf$string || x == tf$bool)
    "nominal"
  else
    "numeric"
}

# Recipe ------------------------------------------------------------------

Recipe <- R6::R6Class(
  "Recipe",
  public = list(
    base_steps = list(),
    derived_steps = list(),
    steps = list(),
    formula = NULL,
    column_names = NULL,
    column_types = NULL,
    column_roles = NULL,
    dataset = NULL,
    fitted = FALSE,
    prepared_dataset = NULL,
    x = NULL,
    y = NULL,

    initialize = function(dataset, x, y = NULL) {
      self$formula <- formula
      self$x <- rlang::enquo(x)
      self$y <- rlang::enquo(y)
      self$set_dataset(dataset)
      self$column_names <- column_names(self$dataset)
      self$column_types <- output_types(self$dataset)
    },

    set_dataset = function(dataset) {
      self$prepared_dataset <- dataset_prepare(dataset, !!self$x, !!self$y, named_features = TRUE)
      self$dataset <- dataset_map(self$prepared_dataset, function(x) x$x)
      invisible(self)
    },

    add_step = function(step) {

      self$steps <- append(self$steps, name_step(step))

    },

    fit = function() {

      if (self$fitted)
        stop("Recipe is already fitted.")

      if (tf$executing_eagerly()) {
        ds <- reticulate::as_iterator(self$dataset)
        nxt <- reticulate::iter_next(ds)
      } else {
        ds <- make_iterator_one_shot(self$dataset)
        nxt_it <- ds$get_next()
        sess <- tf$compat$v1$Session()
        nxt <- sess$run(nxt_it)
      }

      while (!is.null(nxt)) {
        for (i in seq_along(self$steps)) {
          self$steps[[i]]$fit_batch(nxt)
        }

        if (tf$executing_eagerly()) {
          nxt <- reticulate::iter_next(ds)
        } else {
          nxt <- tryCatch({sess$run(nxt_it)}, error = out_of_range_handler)
        }
      }

      for (i in seq_along(self$steps)) {
        self$steps[[i]]$fit_resume()
      }

      self$fitted <- TRUE

      if (!tf$executing_eagerly())
        sess$close()
    },

    features = function() {

      if (!self$fitted)
        stop("Only available after fitting the recipe.")

      feats <- NULL
      for (i in seq_along(self$steps)) {
        stp <- self$steps[i]
        feature <- lapply(stp, function(x) x$feature(feats))
        feats <- append(feats, feature)
        feats <- unlist(feats)
      }

      feats
    },

    dense_features = function() {

      if (!self$fitted)
        stop("Only available after fitting the recipe.")

      Filter(is_dense_column, self$features())
    },

    feature_names = function() {
      unique(c(names(self$steps), self$column_names))
    },

    feature_types = function() {

      feature_names <- self$feature_names()
      feature_types <- character(length = length(feature_names))

      for (i in seq_along(feature_names)) {

        ft <- feature_names[i]

        if (is.null(self$steps[[ft]])) {
          feature_types[i] <- dtype_chr(self$column_types[[which(self$column_names == ft)]])
        } else {
          feature_types[i] <- self$steps[[ft]]$column_type
        }

      }

      feature_types

    }

  ),

  private = list(
    deep_clone = function(name, value) {
      if (inherits(value, "R6")) {
        value$clone(deep = TRUE)
      } else if (name == "steps" || name == "base_steps" ||
                 name == "derived_steps") {
        lapply(value, function(x) x$clone(deep = TRUE))
      } else {
        value
      }
    }
  )
)


# Step --------------------------------------------------------------------

Step <- R6::R6Class(
  classname = "Step",

  public = list(
    name = NULL,

    fit_batch = function (batch) {

    },

    fit_resume = function () {

    }

  ),

  private = list(
    deep_clone = function(name, value) {

      if (inherits(value, "python.builtin.object")) {
        value
      } else if (inherits(value, "R6")) {
        value$clone(deep = TRUE)
      } else {
        value
      }

    }
  )
)

CategoricalStep <- R6::R6Class(
  classname = "CategoricalStep",
  inherit = Step
)

DerivedStep <- R6::R6Class(
  "DerivedStep",
  inherit = Step
)


# StepNumericColumn -------------------------------------------------------

StepNumericColumn <- R6::R6Class(
  "StepNumericColumn",
  inherit = Step,
  public = list(
    key = NULL,
    shape = NULL,
    default_value = NULL,
    dtype = NULL,
    normalizer_fn = NULL,
    column_type = "numeric",
    initialize = function(key, shape, default_value, dtype, normalizer_fn, name) {
      self$key <- key
      self$shape <- shape
      self$default_value <- default_value
      self$dtype <- dtype
      self$normalizer_fn <- normalizer_fn
      self$name <- name
    },
    fit_batch = function(batch) {
      if (is.null(self$normalizer_fn) || is.function(self$normalizer_fn)) {

      } else {

      }
    },
    fit_resume = function() {

    },
    feature = function (base_features) {
      tf$feature_column$numeric_column(
        key = self$key, shape = self$shape,
        default_value = self$default_value,
        dtype = self$dtype,
        normalizer_fn = self$normalizer_fn
      )
    }
  )
)

# StepCategoricalColumnWithVocabularyList ---------------------------------

StepCategoricalColumnWithVocabularyList <- R6::R6Class(
  "StepCategoricalColumnWithVocabularyList",
  inherit = CategoricalStep,
  public = list(

    key = NULL,
    vocabulary_list = NULL,
    dtype = NULL,
    default_value = -1L,
    num_oov_buckets = 0L,
    vocabulary_list_aux = NULL,
    column_type = "nominal",

    initialize = function(key, vocabulary_list = NULL, dtype = NULL, default_value = -1L,
                          num_oov_buckets = 0L, name) {

      self$key <- key
      self$vocabulary_list <- vocabulary_list
      self$dtype = dtype
      self$default_value <- default_value
      self$num_oov_buckets <- num_oov_buckets
      self$name <- name
    },

    fit_batch = function(batch) {

      if (is.null(self$vocabulary_list)) {
        values <- batch[[self$key]]

        if (!is.atomic(values))
          values <- values$numpy()

        unq <- unique(values)
        self$vocabulary_list_aux <- sort(unique(c(self$vocabulary_list_aux, unq)))
      }

    },

    fit_resume = function() {

      if (is.null(self$vocabulary_list)) {
        self$vocabulary_list <- self$vocabulary_list_aux
      }

    },

    feature = function(base_features) {

      tf$feature_column$categorical_column_with_vocabulary_list(
        key = self$key,
        vocabulary_list = self$vocabulary_list,
        dtype = self$dtype,
        default_value = self$default_value,
        num_oov_buckets = self$num_oov_buckets
      )

    }
  )
)


# StepIndicatorColumn -----------------------------------------------------

StepIndicatorColumn <- R6::R6Class(
  "StepIndicatorColumn",
  inherit = Step,
  public = list(
    categorical_column = NULL,
    base_features = NULL,
    column_type = "numeric",
    initialize = function(categorical_column, name) {
      self$categorical_column = categorical_column
      self$name <- name
    },
    feature = function(base_features) {
      tf$feature_column$indicator_column(base_features[[self$categorical_column]])
    }
  )
)


# StepEmbeddingColumn -----------------------------------------------------

StepEmbeddingColumn <- R6::R6Class(
  "StepEmbeddingColumn",
  inherit = Step,
  public = list(

    categorical_column = NULL,
    dimension = NULL,
    combiner = NULL,
    initializer = NULL,
    ckpt_to_load_from = NULL,
    tensor_name_in_ckpt = NULL,
    max_norm = NULL,
    trainable = NULL,
    column_type = "numeric",

    initialize = function(categorical_column, dimension, combiner = "mean", initializer = NULL,
                          ckpt_to_load_from = NULL, tensor_name_in_ckpt = NULL, max_norm = NULL,
                          trainable = TRUE, name) {

      self$categorical_column <- categorical_column
      self$dimension <- dimension
      self$combiner <- combiner
      self$initializer <- initializer
      self$ckpt_to_load_from <- ckpt_to_load_from
      self$tensor_name_in_ckpt <- tensor_name_in_ckpt
      self$max_norm <- max_norm
      self$trainable <- trainable
      self$name <- name

    },

    feature = function(base_features) {

      categorical_column <- base_features[[self$categorical_column]]

      tf$feature_column$embedding_column(
        categorical_column = categorical_column,
        dimension = self$dimension,
        combiner = self$combiner,
        initializer = self$initializer,
        ckpt_to_load_from = self$ckpt_to_load_from,
        tensor_name_in_ckpt = self$tensor_name_in_ckpt,
        max_norm = self$max_norm,
        trainable = self$trainable
      )

    }

  )
)


# StepCrossedColumn -------------------------------------------------------

StepCrossedColumn <- R6::R6Class(
  "StepCrossedColumn",
  inherit = DerivedStep,
  public = list(

    keys = NULL,
    hash_bucket_size = NULL,
    hash_key = NULL,
    column_type = "nominal",

    initialize = function (keys, hash_bucket_size, hash_key = NULL, name = NULL) {
      self$keys <- keys
      self$hash_bucket_size <- hash_bucket_size
      self$hash_key <- hash_key
      self$name <- name
    },

    feature = function(base_features) {

      keys <- lapply(self$keys, function(x) base_features[[x]])
      names(keys) <- NULL

      tf$feature_column$crossed_column(
        keys = keys,
        hash_bucket_size = self$hash_bucket_size,
        hash_key = self$hash_key
      )

    }

  )
)


# StepBucketizedColumn ----------------------------------------------------

StepBucketizedColumn <- R6::R6Class(
  "StepBucketizedColumn",
  inherit = DerivedStep,

  public = list(

    source_column = NULL,
    boundaries = NULL,
    column_type = "numeric",

    initialize = function(source_column, boundaries, name) {
      self$source_column <- source_column
      self$boundaries <- boundaries
      self$name <- name
    },

    feature = function(base_features) {

      tf$feature_column$bucketized_column(
        source_column = base_features[[self$source_column]],
        boundaries = self$boundaries
      )

    }

  )

)


# StepSharedEmbeddings ----------------------------------------------------

StepSharedEmbeddings <- R6::R6Class(
  "StepSharedEmbeddings",
  inherit = DerivedStep,
  public = list(
    categorical_columns = NULL,
    dimension = NULL,
    combiner = NULL,
    initializer = NULL,
    shared_embedding_collection_name = NULL,
    ckpt_to_load_from = NULL,
    tensor_name_in_ckpt = NULL,
    max_norm = NULL,
    trainable = NULL,
    column_type = "numeric",

    initialize = function(categorical_columns, dimension, combiner = "mean",
                              initializer = NULL, shared_embedding_collection_name = NULL,
                              ckpt_to_load_from = NULL, tensor_name_in_ckpt = NULL,
                              max_norm = NULL, trainable = TRUE, name = NULL) {
      self$categorical_columns <- categorical_columns
      self$dimension <- dimension
      self$combiner <- combiner
      self$initializer <- initializer
      self$shared_embedding_collection_name <- shared_embedding_collection_name
      self$ckpt_to_load_from <- ckpt_to_load_from
      self$tensor_name_in_ckpt <- tensor_name_in_ckpt
      self$max_norm <- max_norm
      self$trainable <- trainable
      self$name <- name
    },

    feature = function(base_features) {
      categorical_columns <- lapply(self$categorical_columns, function(x) {
        base_features[[x]]
      })
      names(categorical_columns) <- NULL

      tf$feature_column$shared_embeddings(
        categorical_columns = categorical_columns,
        dimension = self$dimension,
        combiner = self$combiner,
        initializer = self$initializer,
        shared_embedding_collection_name = self$shared_embedding_collection_name,
        ckpt_to_load_from = self$ckpt_to_load_from,
        tensor_name_in_ckpt = self$tensor_name_in_ckpt,
        max_norm = self$max_norm,
        trainable = self$trainable
      )
    }
  )
)


# Wrappers ----------------------------------------------------------------

#' @export
recipe <- function(dataset, x, y = NULL) {
  en_x <- rlang::enquo(x)
  en_y <- rlang::enquo(y)
  rec <- Recipe$new(dataset, x = !!en_x, y = !!en_y)
  rec
}

#' @export
prep <- function(rec, dataset=NULL) {
  rec <- rec$clone(deep = TRUE)

  if (!is.null(dataset))
    rec$set_dataset(dataset)

  rec$fit()
  rec
}

#' @export
juice <- function(rec) {
  if (reticulate::py_is_null_xptr(rec$prepared_dataset))
    stop("The tensorflow dataset was not found. Please use `bake` instead.")

  if (!rec$fitted)
    stop("Recipe must be prepared before juicing.")

  rec$prepared_dataset
}

#' @export
bake <- function(rec, dataset) {

  if (!inherits(dataset, "tensorflow.python.data.ops.dataset_ops.DatasetV2"))
    stop("`dataset` must be a TensorFlow dataset.")

  if (!rec$fitted)
    stop("Recipe must be prepared before juicing.")

  rec <- rec$clone(deep = TRUE)
  rec$set_dataset(dataset)
  rec$prepared_dataset
}


#' @export
step_numeric_column <- function(rec, ..., shape = 1L, default_value = NULL,
                                dtype = tf$float32, normalizer_fn = NULL) {

  rec <- rec$clone(deep = TRUE)
  variables <- tidyselect::vars_select(rec$feature_names(), !!!quos(...))
  for (var in variables) {
    stp <- StepNumericColumn$new(var, shape, default_value, dtype, normalizer_fn,
                                 name = var)
    rec$add_step(stp)
  }

  rec
}

#' @export
step_categorical_column_with_vocabulary_list <- function(rec, ..., vocabulary_list = NULL,
                                                         dtype = NULL, default_value = -1L,
                                                         num_oov_buckets = 0L) {
  rec <- rec$clone(deep = TRUE)
  variables <- tidyselect::vars_select(rec$feature_names(), !!!quos(...))
  for (var in variables) {
    stp <- StepCategoricalColumnWithVocabularyList$new(
      var, vocabulary_list, dtype,
      default_value, num_oov_buckets,
      name = var
    )
    rec$add_step(stp)
  }

  rec
}

make_step_name <- function(quosure, variable, step) {
  nms <- names(quosure)
  if (!is.null(nms) && nms != "") {
    nms
  } else {
    paste0(step, "_", variable)
  }
}

#' @export
step_indicator_column <- function(rec, ...) {

  rec <- rec$clone(deep = TRUE)
  quosures <- quos(...)

  variables <- tidyselect::vars_select(rec$feature_names(), !!!quosures)

  for (i in seq_along(quosures)) {

    stp <- StepIndicatorColumn$new(
      variables[i],
      name = make_step_name(quosures[i], variables[i], "indicator")
    )
    rec$add_step(stp)
  }

  rec
}

#' @export
step_embedding_column <- function(rec, ..., dimension, combiner = "mean",
                                  initializer = NULL, ckpt_to_load_from = NULL,
                                  tensor_name_in_ckpt = NULL, max_norm = NULL,
                                  trainable = TRUE) {
  rec <- rec$clone(deep = TRUE)
  quosures <- quos(...)

  variables <- tidyselect::vars_select(rec$feature_names(), !!!quosures)

  for (i in seq_along(variables)) {
    stp <- StepEmbeddingColumn$new(
      variables[i], dimension, combiner, initializer,
      ckpt_to_load_from, tensor_name_in_ckpt,
      max_norm, trainable,
      name = make_step_name(quosures[i], variables[i], "embedding")
    )
    rec$add_step(stp)
  }

  rec
}


make_multiple_columns_step_name <- function(quosure, variables, step) {
  nms <- names(quosure)
  if (!is.null(nms) && nms != "") {
    nms
  } else {
    paste0(step, "_", paste(variables, collapse= "_"))
  }
}

#' @export
step_crossed_column <- function(rec, ..., hash_bucket_size, hash_key = NULL) {

  rec <- rec$clone(deep = TRUE)
  quosures <- quos(...)


  for (i in seq_along(quosures)) {

    variables <- tidyselect::vars_select(rec$feature_names(), !!quosures[[i]])

    stp <- StepCrossedColumn$new(
      keys = variables,
      hash_bucket_size = hash_bucket_size,
      hash_key = hash_key,
      name = make_multiple_columns_step_name(quosures[i], variables, "crossed")
    )

    rec$add_step(stp)
  }

  rec

}

#' @export
step_bucketized_column <- function(rec, ..., boundaries) {

  rec <- rec$clone(deep = TRUE)
  quosures <- quos(...)

  variables <- tidyselect::vars_select(rec$feature_names(), !!!quosures)

  for (i in seq_along(variables)) {

    stp <- StepBucketizedColumn$new(
      variables[i],
      boundaries = boundaries,
      name = make_step_name(quosures[i], variables[i], "bucketized")
    )
    rec$add_step(stp)
  }

  rec

}

#' @export
step_shared_embeddings_column <- function(rec, ..., dimension, combiner = "mean",
                                          initializer = NULL, shared_embedding_collection_name = NULL,
                                          ckpt_to_load_from = NULL, tensor_name_in_ckpt = NULL,
                                          max_norm = NULL, trainable = TRUE) {
  rec <- rec$clone(deep = TRUE)
  quosures <- quos(...)

  for (i in seq_along(quosures)) {
    variables <- tidyselect::vars_select(rec$feature_names(), !!quosures[[i]])

    stp <- StepSharedEmbeddings$new(
      categorical_columns = variables,
      dimension = dimension,
      combiner = combiner,
      initializer = initializer,
      shared_embedding_collection_name = shared_embedding_collection_name,
      ckpt_to_load_from = ckpt_to_load_from,
      tensor_name_in_ckpt = tensor_name_in_ckpt,
      max_norm = max_norm,
      trainable = trainable,
      name = make_multiple_columns_step_name(quosures[i], variables, "shared_embeddings")
    )

    rec$add_step(stp)
  }

  rec
}
