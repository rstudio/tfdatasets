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
  x$name
}

# Selectors ---------------------------------------------------------------

#' Selectors
#'
#' List of selectors that can be used to specify variables inside
#' steps.
#'
#' @section Selectors:
#'
#' * [has_type()]
#' * [all_numeric()]
#' * [all_nominal()]
#'
#' @name selectors
#' @rdname selectors

cur_info_env <- rlang::child_env(rlang::env_parent(rlang::env()))

set_current_info <- function(x) {
  old <- cur_info_env
  cur_info_env$feature_names <- x$feature_names
  cur_info_env$feature_types <- x$feature_types

  invisible(old)
}

current_info <- function() {
  cur_info_env %||% stop("Variable context not set", call. = FALSE)
}

#' Identify the type of the variable.
#'
#' Can only be used inside the [steps] specifications to find
#' variables by type.
#'
#' @param match A list of types to match.
#'
#' @family Selectors
#' @export
has_type <- function(match = "float32") {
  info <- current_info()
  lgl_matches <- purrr::map_lgl(info$feature_types, ~any(.x %in% match))
  info$feature_names[which(lgl_matches)]
}

terms_select <- function(feature_names, feature_types, terms) {

  old_info <- set_current_info(
    list(feature_names = feature_names, feature_types = feature_types)
    )
  on.exit(set_current_info(old_info), add = TRUE)

  sel <- tidyselect::vars_select(feature_names, !!! terms)

  sel
}

#' Speciy all numeric variables.
#'
#' Find all the variables with the following types:
#' "float16", "float32", "float64", "int16", "int32", "int64",
#' "half", "double".
#'
#' @family Selectors
#' @export
all_numeric <- function() {
  has_type(c("float16", "float32", "float64", "int16", "int32", "int64", "half", "double"))
}

#' Find all nominal variables.
#'
#' Currently we only consider "string" type as nominal.
#'
#' @family Selectors
#' @export
all_nominal <- function() {
  has_type(c("string"))
}

# FeatureSpec ------------------------------------------------------------------

FeatureSpec <- R6::R6Class(
  "FeatureSpec",
  public = list(
    steps = list(),
    formula = NULL,
    column_names = NULL,
    column_types = NULL,
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
        stop("FeatureSpec is already fitted.")

      if (tf$executing_eagerly()) {
        ds <- reticulate::as_iterator(self$dataset)
        nxt <- reticulate::iter_next(ds)
      } else {
        ds <- make_iterator_one_shot(self$dataset)
        nxt_it <- ds$get_next()
        sess <- tf$compat$v1$Session()
        nxt <- sess$run(nxt_it)
      }

      pb <- progress::progress_bar$new(
        format = ":spin Preparing :tick_rate batches/s [:current batches in :elapsedfull]",
        total = Inf)

      while (!is.null(nxt)) {
        pb$tick(1)
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
        stop("Only available after fitting the feature_spec.")

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
        stop("Only available after fitting the feature_spec.")

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
        } else if (is.null(self$steps[[ft]]$column_type)) {
          feature_types[i] <- dtype_chr(self$column_types[[which(self$column_names == ft)]])
        } else {
          feature_types[i] <- self$steps[[ft]]$column_type
        }

      }

      feature_types

    },

    print = function() {
      cat(cli::style_bold(paste("A feature_spec with", length(self$steps), "steps.\n")))

      cat("Prepared:", self$fitted, "\n")

      if (self$fitted)
        cat("The feature_spec has", length(self$dense_features), "dense features.\n")

      if (length(self$steps) > 0) {
        step_types <- sapply(self$steps, function(x) class(x)[1])
        for (step_type in sort(unique(step_types))) {
          cat(paste0(cli::style_bold(step_type), ":"), paste(names(step_types[step_types == step_type]), collapse = ", "), "\n")
        }
      }
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

# Scalers -----------------------------------------------------------------

Normalizer <- R6::R6Class(
  "Normalizer",
  public = list(
    fit_batch = function(batch) {

    },
    fit_resume = function() {

    },
    fun = function() {

    }
  )
)

StandardScaler <- R6::R6Class(
  "StandardScaler",
  inherit = Normalizer,
  public = list(
    sum = 0,
    sum_2 = 0,
    n = 0,
    sd = NULL,
    mean = NULL,
    fit_batch = function (batch) {
      self$sum <- self$sum + sum(batch)
      self$sum_2 <- self$sum_2 + sum(batch^2)
      self$n <- self$n + length(batch)
    },
    fit_resume = function() {
      self$mean <- self$sum/self$n
      self$sd <- sqrt((self$n/(self$n-1))*(self$sum_2/self$n - (self$sum/self$n)^2))
    },
    fun = function() {
      mean_ <- self$mean
      sd_ <- self$sd
      function(x) {

        if (!x$dtype$is_floating)
          x <- tf$cast(x, tf$float32)

        (x - mean_)/sd_

      }
    }
  )
)

#' @export
standard_scaler <- function() {
  StandardScaler$new()
}

MinMaxScaler <- R6::R6Class(
  "MinMaxScaler",
  inherit = Normalizer,
  public = list(
    min = Inf,
    max = -Inf,
    fit_batch = function (batch) {
      self$min <- min(c(self$min, min(batch)))
      self$max <- max(c(self$max, max(batch)))
    },
    fun = function() {
      min_ <- self$min
      max_ <- self$max
      function(x) {

        if (!x$dtype$is_floating)
          x <- tf$cast(x, tf$float32)

        (x - min_)/(max_ - min_)

      }
    }
  )
)

#' @export
scaler_standard <- function() {
  StandardScaler$new()
}

#' @export
scaler_min_max <- function() {
  MinMaxScaler$new()
}

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
    column_type = NULL,
    initialize = function(key, shape, default_value, dtype, normalizer_fn, name) {
      self$key <- key
      self$shape <- shape
      self$default_value <- default_value
      self$dtype <- dtype
      self$normalizer_fn <- normalizer_fn
      self$name <- name
      self$column_type = dtype_chr(dtype)
    },
    fit_batch = function(batch) {
      if (inherits(self$normalizer_fn, "Normalizer")) {
        self$normalizer_fn$fit_batch(as.numeric(batch[[self$key]]))
      }
    },
    fit_resume = function() {
      if (inherits(self$normalizer_fn, "Normalizer")) {
        self$normalizer_fn$fit_resume()
        self$normalizer_fn <- self$normalizer_fn$fun()
      }
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
    column_type = NULL,

    initialize = function(key, vocabulary_list = NULL, dtype = NULL, default_value = -1L,
                          num_oov_buckets = 0L, name) {

      self$key <- key
      self$vocabulary_list <- vocabulary_list
      self$dtype = dtype
      self$default_value <- default_value
      self$num_oov_buckets <- num_oov_buckets
      self$name <- name
      if (!is.null(dtype)) {
        self$column_type = dtype_chr(dtype)
      }
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


# StepCategoricalColumnWithHashBucket -------------------------------------

StepCategoricalColumnWithHashBucket <- R6::R6Class(
  "StepCategoricalColumnWithHashBucket",
  inherit = CategoricalStep,
  public = list(
    key = NULL,
    hash_bucket_size = NULL,
    dtype = NULL,
    column_type = NULL,
    initialize = function(key, hash_bucket_size, dtype = tf$string, name) {
      self$key <- key
      self$hash_bucket_size <- hash_bucket_size
      self$dtype <- dtype
      self$name <- name
      if (!is.null(dtype)) {
        self$column_type = dtype_chr(dtype)
      }
    },
    feature = function (base_features) {
      tf$feature_column$categorical_column_with_hash_bucket(
        key = self$key,
        hash_bucket_size = self$hash_bucket_size,
        dtype = self$dtype
      )
    }
  )
)

# StepCategoricalColumnWithIdentity -------------------------------------

StepCategoricalColumnWithIdentity <- R6::R6Class(
  "StepCategoricalColumnWithIdentity",
  inherit = CategoricalStep,
  public = list(
    key = NULL,
    num_buckets = NULL,
    default_value = NULL,
    initialize = function(key, num_buckets, default_value = NULL, name) {
      self$key <- key
      self$num_buckets <- num_buckets
      self$default_value <- default_value
      self$name <- name
    },
    feature = function (base_features) {
      tf$feature_column$categorical_column_with_identity(
        key = self$key,
        num_buckets = self$num_buckets,
        default_value = self$default_value
      )
    }
  )
)

# StepCategoricalColumnWithVocabularyFile -------------------------------------

StepCategoricalColumnWithVocabularyFile <- R6::R6Class(
  "StepCategoricalColumnWithVocabularyFile",
  inherit = CategoricalStep,
  public = list(
    key = NULL,
    vocabulary_file = NULL,
    vocabulary_size = NULL,
    dtype = NULL,
    default_value = NULL,
    num_oov_buckets = NULL,
    column_type = NULL,
    initialize = function(key, vocabulary_file, vocabulary_size = NULL, dtype = tf$string,
                          default_value = NULL, num_oov_buckets = 0L, name) {
      self$key <- key
      self$vocabulary_file <- normalizePath(vocabulary_file)
      self$vocabulary_size <- vocabulary_size
      self$dtype <- dtype
      self$default_value <- default_value
      self$num_oov_buckets <- num_oov_buckets
      self$name <- name
      if (!is.null(dtype)) {
        self$column_type = dtype_chr(dtype)
      }
    },
    feature = function (base_features) {
      tf$feature_column$categorical_column_with_vocabulary_file(
        key = self$key,
        vocabulary_file = self$vocabulary_file,
        vocabulary_size = self$vocabulary_size,
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
    column_type = "float32",
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
    column_type = "float32",

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
    column_type = "string",

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
    column_type = "float32",

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
    column_type = "float32",

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

#' Creates a feature specification.
#'
#' Used to create initilialize a feature columns specification.
#'
#' @param dataset A TensorFlow dataset.
#' @param x Features to include can use [tidyselect::select_helpers()] or
#'   a `formula`.
#' @param y (Optional) The response variable. Can also be specified using
#'   a `formula` in the `x` argument.
#'
#' @details
#' After creating the `feature_spec` object you can add steps using the
#' `step` functions.
#'
#' @return a `FeatureSpec` object.
#'
#' @seealso
#' * [fit.FeatureSpec()] to fit the FeatureSpec
#' * [dataset_use_spec()] to create a tensorflow dataset prepared to modeling.
#' * [steps] to a list of all implemented steps.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ .)
#'
#' # select using `tidyselect` helpers
#' spec <- feature_spec(hearts, x = c(thal, age), y = target)
#' }
#' @family Feature Spec Functions
#' @export
feature_spec <- function(dataset, x, y = NULL) {
  en_x <- rlang::enquo(x)
  en_y <- rlang::enquo(y)
  spec <- FeatureSpec$new(dataset, x = !!en_x, y = !!en_y)
  spec
}

#' Fits a feature specification.
#'
#' This function will `fit` the specification. Depending
#' on the steps added to the specification it will compute
#' for example, the levels of categorical features, normalization
#' constants, etc.
#'
#' @param spec A feature specification created with [feature_spec()].
#' @param dataset (Optional) A TensorFlow dataset. If `NULL` it will use
#'   the dataset provided when initilializing the `feature_spec`.
#' @param ... (unused)
#'
#' @seealso
#' * [feature_spec()] to initialize the feature specification.
#' * [dataset_use_spec()] to create a tensorflow dataset prepared to modeling.
#' * [steps] to a list of all implemented steps.
#'
#' @return a fitted `FeatureSpec` object.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ age) %>%
#'   step_numeric_column(age)
#'
#' spec_fit <- fit(spec)
#' spec_fit
#' }
#' @family Feature Spec Functions
#' @export
fit.FeatureSpec <- function(spec, dataset=NULL, ...) {
  spec <- spec$clone(deep = TRUE)

  if (!is.null(dataset))
    spec$set_dataset(dataset)

  spec$fit()
  spec
}

#' Transform the dataset using the provided spec.
#'
#' Prepares the dataset to be used directly in a model.The transformed
#' dataset is prepared to return tuples (x,y) that can be used directly
#' in Keras.
#'
#' @param dataset A TensorFlow dataset.
#' @param spec A feature specification created with [feature_spec()].
#' @seealso
#' * [feature_spec()] to initialize the feature specification.
#' * [fit.FeatureSpec()] to create a tensorflow dataset prepared to modeling.
#' * [steps] to a list of all implemented steps.
#'
#' @return A TensorFlow dataset.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ age) %>%
#'   step_numeric_column(age)
#'
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(hearts, spec_fit)
#' }
#' @family Feature Spec Functions
#' @export
dataset_use_spec <- function(dataset, spec) {

  if (!inherits(dataset, "tensorflow.python.data.ops.dataset_ops.DatasetV2"))
    stop("`dataset` must be a TensorFlow dataset.")

  if (!spec$fitted)
    stop("FeatureSpec must be prepared before juicing.")

  spec <- spec$clone(deep = TRUE)
  spec$set_dataset(dataset)
  spec$prepared_dataset %>%
    dataset_map(function(x) reticulate::tuple(x$x, x$y))
}

#' Steps for feature columns specification.
#'
#' List of steps that can be used to specify columns in the `feature_spec` interface.
#'
#' @section Steps:
#'
#' * [step_numeric_column()] to define numeric columns.
#' * [step_categorical_column_with_vocabulary_list()] to define categorical columns.
#' * [step_categorical_column_with_hash_bucket()]
#' * [step_categorical_column_with_identity()]
#' * [step_categorical_column_with_vocabulary_file()]
#' * [step_indicator_column()]
#' * [step_embedding_column()]
#' * [step_bucketized_column()]
#' * [step_crossed_column()]
#' * [step_shared_embeddings_column()]
#'
#' @seealso
#' * [selectors] for a list of selectors that can be used to specify variables.
#'
#' @name steps
#' @rdname steps
#' @family Feature Spec Functions
NULL

#' @family Feature Spec Functions
#' @export
step_numeric_column <- function(spec, ..., shape = 1L, default_value = NULL,
                                dtype = tf$float32, normalizer_fn = NULL) {

  spec <- spec$clone(deep = TRUE)
  quos_ <- quos(...)

  variables <- terms_select(spec$feature_names(), spec$feature_types(), quos_)
  for (var in variables) {
    stp <- StepNumericColumn$new(var, shape, default_value, dtype, normalizer_fn,
                                 name = var)
    spec$add_step(stp)
  }

  spec
}

#' @export
step_categorical_column_with_vocabulary_list <- function(spec, ..., vocabulary_list = NULL,
                                                         dtype = NULL, default_value = -1L,
                                                         num_oov_buckets = 0L) {
  spec <- spec$clone(deep = TRUE)
  quos_ <- quos(...)

  variables <- terms_select(spec$feature_names(), spec$feature_types(), quos_)
  for (var in variables) {
    stp <- StepCategoricalColumnWithVocabularyList$new(
      var, vocabulary_list, dtype,
      default_value, num_oov_buckets,
      name = var
    )
    spec$add_step(stp)
  }

  spec
}

#' @export
step_categorical_column_with_hash_bucket <- function(spec, ..., hash_bucket_size,
                                                     dtype = tf$string) {

  spec <- spec$clone(deep = TRUE)
  quos_ <- quos(...)

  variables <- terms_select(spec$feature_names(), spec$feature_types(), quos_)
  for (var in variables) {
    stp <- StepCategoricalColumnWithHashBucket$new(
      var,
      hash_bucket_size = hash_bucket_size,
      dtype = dtype,
      name = var
    )
    spec$add_step(stp)
  }

  spec
}

#' @export
step_categorical_column_with_identity <- function(spec, ..., num_buckets,
                                                     default_value = NULL) {

  spec <- spec$clone(deep = TRUE)
  quos_ <- quos(...)

  variables <- terms_select(spec$feature_names(), spec$feature_types(), quos_)
  for (var in variables) {
    stp <- StepCategoricalColumnWithIdentity$new(
      key = var,
      num_buckets = num_buckets,
      default_value = default_value,
      name = var
    )
    spec$add_step(stp)
  }

  spec
}

#' @export
step_categorical_column_with_vocabulary_file <- function(spec, ..., vocabulary_file,
                                                         vocabulary_size = NULL,
                                                         dtype = tf$string,
                                                         default_value = NULL,
                                                         num_oov_buckets = 0L) {
  spec <- spec$clone(deep = TRUE)
  quos_ <- quos(...)

  variables <- terms_select(spec$feature_names(), spec$feature_types(), quos_)
  for (var in variables) {
    stp <- StepCategoricalColumnWithVocabularyFile$new(
      key = var,
      vocabulary_file = vocabulary_file,
      vocabulary_size = vocabulary_size,
      dtype = dtype,
      default_value = default_value,
      num_oov_buckets = num_oov_buckets,
      name = var
    )
    spec$add_step(stp)
  }

  spec
}

make_step_name <- function(quosure, variable, step) {

  nms <- names(quosure)

  if (!is.null(nms) && !is.na(nms) && length(nms) == 1 && nms != "" ) {
    nms
  } else {
    paste0(step, "_", variable)
  }
}

step_ <- function(spec, ..., step, args, prefix) {

  spec <- spec$clone(deep = TRUE)

  quosures <- quos(...)
  variables <- terms_select(spec$feature_names(), spec$feature_types(), quosures)
  nms <- names(quosures)

  if ( !is.null(nms) && any(nms != "") && length(nms) != length(variables) )
    stop("Can't name feature if using a selector.")

  for (i in seq_along(variables)) {

    args_ <- append(
      list(
        variables[i],
        name = make_step_name(quosures[i], variables[i], prefix)
      ),
      args
    )

    stp <- do.call(step, args_)

    spec$add_step(stp)
  }


  spec
}

#' @export
step_indicator_column <- function(spec, ...) {
  step_(spec, ..., step = StepIndicatorColumn$new, args = list(), prefix = "indicator")
}

#' @export
step_embedding_column <- function(spec, ..., dimension, combiner = "mean",
                                  initializer = NULL, ckpt_to_load_from = NULL,
                                  tensor_name_in_ckpt = NULL, max_norm = NULL,
                                  trainable = TRUE) {


  args <- list(
    dimension = dimension,
    combiner = combiner,
    initializer = initializer,
    ckpt_to_load_from = ckpt_to_load_from,
    tensor_name_in_ckpt = tensor_name_in_ckpt,
    max_norm = max_norm,
    trainable = trainable
  )

  step_(spec, ..., step = StepEmbeddingColumn$new, args = args, prefix = "embedding")
}

#' @export
step_bucketized_column <- function(spec, ..., boundaries) {
  args <- list(
    boundaries = boundaries
  )

  step_(spec, ..., step = StepBucketizedColumn$new, args = args, prefix = "bucketized")
}

make_multiple_columns_step_name <- function(quosure, variables, step) {
  nms <- names(quosure)
  if (!is.null(nms) && nms != "") {
    nms
  } else {
    paste0(step, "_", paste(variables, collapse= "_"))
  }
}

step_multiple_ <- function(spec, ..., step, args, prefix) {

  spec <- spec$clone(deep = TRUE)
  quosures <- quos(...)


  for (i in seq_along(quosures)) {

    variables <- terms_select(spec$feature_names(), spec$feature_types(), quosures[i])
    args_ <- append(
      list(
        variables,
        name = make_multiple_columns_step_name(quosures[i], variables, "crossed")
      ),
      args
    )

    stp <- do.call(step, args_)

    spec$add_step(stp)
  }

  spec


}

#' @export
step_crossed_column <- function(spec, ..., hash_bucket_size, hash_key = NULL) {

  args <- list(
    hash_bucket_size = hash_bucket_size,
    hash_key = hash_key
  )

  step_multiple_(spec, ..., step = StepCrossedColumn$new, args = args, prefix = "crossed")
}

#' @export
step_shared_embeddings_column <- function(spec, ..., dimension, combiner = "mean",
                                          initializer = NULL, shared_embedding_collection_name = NULL,
                                          ckpt_to_load_from = NULL, tensor_name_in_ckpt = NULL,
                                          max_norm = NULL, trainable = TRUE) {

  args <- list(
    dimension = dimension,
    combiner = combiner,
    initializer = initializer,
    shared_embedding_collection_name = shared_embedding_collection_name,
    ckpt_to_load_from = ckpt_to_load_from,
    tensor_name_in_ckpt = tensor_name_in_ckpt,
    max_norm = max_norm,
    trainable = trainable
  )

  step_multiple_(
    spec, ...,
    step = StepSharedEmbeddings$new,
    args = args,
    prefix = "shared_embeddings"
  )
}
