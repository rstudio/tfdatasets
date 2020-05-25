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
#' * [starts_with()]
#' * [ends_with()]
#' * [one_of()]
#' * [matches()]
#' * [contains()]
#' * [everything()]
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
  lgl_matches <- sapply(info$feature_types, function(x) any(x %in% match))
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

#' @importFrom tidyselect starts_with
#' @export
tidyselect::starts_with

#' @importFrom tidyselect ends_with
#' @export
tidyselect::ends_with

#' @importFrom tidyselect contains
#' @export
tidyselect::contains

#' @importFrom tidyselect everything
#' @export
tidyselect::everything

#' @importFrom tidyselect matches
#' @export
tidyselect::matches

#' @importFrom tidyselect num_range
#' @export
tidyselect::num_range

#' @importFrom tidyselect one_of
#' @export
tidyselect::one_of

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

      if (inherits(dataset, "data.frame")) {
        dataset <- tensors_dataset(dataset)
      }

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

        if (inherits(stp[[1]], "RemoveStep")) {
          feats <- feats[-which(names(feats) == stp[[1]]$var)]
        } else {
          feature <- lapply(stp, function(x) x$feature(feats)) # keep list names
          feats <- append(feats, feature)
          feats <- unlist(feats)
        }
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
      cat(cli::rule(left = "Feature Spec"), "\n")
      cat(cli::style_bold(paste("A feature_spec with", length(self$steps), "steps.\n")))

      cat(cli::style_bold("Fitted:"), self$fitted, "\n")
      cat(cli::rule(left = "Steps"), "\n")

      if (self$fitted)
        cat("The feature_spec has", length(self$dense_features), "dense features.\n")

      if (length(self$steps) > 0) {
        step_types <- sapply(self$steps, function(x) class(x)[1])
        for (step_type in sort(unique(step_types))) {
          cat(
            paste0(cli::style_bold(step_type), ":"),
            paste(
              names(step_types[step_types == step_type]),
              collapse = ", "
            ),
            "\n"
          )
        }
      }

      cat(cli::rule(left = "Dense features"), "\n")
      if (self$fitted) {



      } else {
        cat("Feature spec must be fitted before we can detect the dense features.\n")
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


RemoveStep <- R6::R6Class(
  "RemoveStep",
  inherit = Step,

  public = list(

    var = NULL,

    initialize = function(var) {
      self$var <- var
      self$name <- var
    }

  )
)

DerivedStep <- R6::R6Class(
  "DerivedStep",
  inherit = Step
)

# Scalers -----------------------------------------------------------------

Scaler <- R6::R6Class(
  "Scaler",
  public = list(
    fit_batch = function(batch) {

    },
    fit_resume = function() {

    },
    fun = function() {

    }
  )
)

# http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
# batch updates for mean and variance.
StandardScaler <- R6::R6Class(
  "StandardScaler",
  inherit = Scaler,
  public = list(
    m = 0,
    sd = 0,
    mean = 0,
    fit_batch = function (batch) {
      m <- self$m
      mu_m <- self$mean
      sd_m <- self$sd

      n <- length(batch)
      mu_n <- mean(batch)
      sd_n <- sqrt(var(batch)*(n-1)/(n))

      self$mean <- (m*mu_m + n*mu_n)/(n + m)
      self$sd <- sqrt((m*(sd_m^2) + n*(sd_n^2))/(m+n) + m*n/((m+n)^2)*((mu_m - mu_n)^2))
      self$m <- m + n
    },
    fit_resume = function() {
      self$sd <- sqrt((self$sd^2)*self$m/(self$m -1))
    },
    fun = function() {
      mean_ <- self$mean
      sd_ <- self$sd
      function(x) {

        if (!x$dtype$is_floating)
          x <- tf$cast(x, tf$float32)

        (x - tf$cast(mean_, x$dtype))/tf$cast(sd_, x$dtype)

      }
    }
  )
)

MinMaxScaler <- R6::R6Class(
  "MinMaxScaler",
  inherit = Scaler,
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

        (x - tf$cast(min_, x$dtype))/(tf$cast(max_, x$dtype) - tf$cast(min_, x$dtype))

      }
    }
  )
)

#' List of pre-made scalers
#'
#' * [scaler_standard]: mean and standard deviation normalizer.
#' * [scaler_min_max]: min max normalizer
#'
#' @seealso [step_numeric_column]
#' @name scaler
#' @rdname scaler
NULL

#' Creates an instance of a standard scaler
#'
#' This scaler will learn the mean and the standard deviation
#' and use this to create a `normalizer_fn`.
#'
#' @seealso [scaler] to a complete list of normalizers
#' @family scaler
#' @export
scaler_standard <- function() {
  StandardScaler$new()
}

#' Creates an instance of a min max scaler
#'
#' This scaler will learn the min and max of the numeric variable
#' and use this to create a `normalizer_fn`.
#'
#' @seealso [scaler] to a complete list of normalizers
#' @family scaler
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
      if (inherits(self$normalizer_fn, "Scaler")) {
        self$normalizer_fn$fit_batch(as.numeric(batch[[self$key]]))
      }
    },
    fit_resume = function() {
      if (inherits(self$normalizer_fn, "Scaler")) {
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

        if (inherits(values, "tensorflow.tensor")) {
          # add shape to tensor with no shape
          if (identical(values$shape$as_list(), list()))
            values <- tf$constant(values, shape = 1L)

          # get unique values before converting to R.
          values <- tensorflow::tf$unique(values)$y

          if (!is.atomic(values))
            values <- values$numpy()
        }

        # converts from bytes to an R string. Need in python >= 3.6
        # special case when values is a single value of type string
        if (inherits(values, "python.builtin.bytes"))
          values <- values$decode()

        if (inherits(values[[1]], "python.builtin.bytes"))
          values <- sapply(values, function(x) x$decode())

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

    initialize = function(categorical_column, dimension = NULL, combiner = "mean", initializer = NULL,
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

      if (is.function(self$dimension)) {
        dimension <- self$dimension(length(categorical_column$vocabulary_list))
      } else {
        dimension <- self$dimension
      }

      tf$feature_column$embedding_column(
        categorical_column = categorical_column,
        dimension = as.integer(dimension),
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
#' Used to create initialize a feature columns specification.
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
  # currently due to a bug in TF we are only using feature columns api with TF
  # >= 2.0. see https://github.com/tensorflow/tensorflow/issues/30307
  if (tensorflow::tf_version() < "2.0")
    stop("Feature spec is only available with TensorFlow >= 2.0", call. = FALSE)
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
#' @param object A feature specification created with [feature_spec()].
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
fit.FeatureSpec <- function(object, dataset=NULL, ...) {
  spec <- object$clone(deep = TRUE)

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
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
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
#' * [step_categorical_column_with_hash_bucket()] to define categorical columns
#'   where ids are set by hashing.
#' * [step_categorical_column_with_identity()] to define categorical columns
#'   represented by integers in the range `[0-num_buckets)`.
#' * [step_categorical_column_with_vocabulary_file()] to define categorical columns
#'   when their vocabulary is available in a file.
#' * [step_indicator_column()] to create indicator columns from categorical columns.
#' * [step_embedding_column()] to create embeddings columns from categorical columns.
#' * [step_bucketized_column()] to create bucketized columns from numeric columns.
#' * [step_crossed_column()] to perform crosses of categorical columns.
#' * [step_shared_embeddings_column()] to share embeddings between a list of
#'   categorical columns.
#' * [step_remove_column()] to remove columns from the specification.
#'
#' @seealso
#' * [selectors] for a list of selectors that can be used to specify variables.
#'
#' @name steps
#' @rdname steps
#' @family Feature Spec Functions
NULL


#' Creates a numeric column specification
#'
#' `step_numeric_column` creates a numeric column specification. It can also be
#' used to normalize numeric columns.
#'
#' @param spec A feature specification created with [feature_spec()].
#' @param ... Comma separated list of variable names to apply the step. [selectors] can also be used.
#' @param shape An iterable of integers specifies the shape of the Tensor. An integer can be given
#'   which means a single dimension Tensor with given width. The Tensor representing the column will
#'   have the shape of `batch_size` + `shape`.
#' @param default_value A single value compatible with `dtype` or an iterable of values compatible
#'   with `dtype` which the column takes on during `tf.Example` parsing if data is missing. A
#'   default value of `NULL` will cause `tf.parse_example` to fail if an example does not contain
#'   this column. If a single value is provided, the same value will be applied as
#'   the default value for every item. If an iterable of values is provided, the shape
#'   of the default_value should be equal to the given shape.
#' @param dtype defines the type of values. Default value is `tf$float32`. Must be a non-quantized,
#'   real integer or floating point type.
#' @param normalizer_fn If not `NULL`, a function that can be used to normalize the value
#'   of the tensor after default_value is applied for parsing. Normalizer function takes the
#'   input Tensor as its argument, and returns the output Tensor. (e.g. `function(x) (x - 3.0) / 4.2)`.
#'   Please note that even though the most common use case of this function is normalization, it
#'   can be used for any kind of Tensorflow transformations. You can also a pre-made [scaler], in
#'   this case a function will be created after [fit.FeatureSpec] is called on the feature specification.
#'
#' @return a `FeatureSpec` object.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ age) %>%
#'   step_numeric_column(age, normalizer_fn = standard_scaler())
#'
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
#' }
#'
#' @seealso [steps] for a complete list of allowed steps.
#'
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

#' Creates a step that can remove columns
#'
#' Removes features of the feature specification.
#'
#' @inheritParams step_numeric_column
#'
#' @return a `FeatureSpec` object.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ age) %>%
#'   step_numeric_column(age, normalizer_fn = scaler_standard()) %>%
#'   step_bucketized_column(age, boundaries = c(20, 50)) %>%
#'   step_remove_column(age)
#'
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
#' }
#'
#' @seealso [steps] for a complete list of allowed steps.
#' @family Feature Spec Functions
#'
#' @export
step_remove_column <- function(spec, ...) {

  spec <- spec$clone(deep = TRUE)
  quos_ <- quos(...)

  variables <- terms_select(spec$feature_names(), spec$feature_types(), quos_)

  for (var in variables) {
    stp <- RemoveStep$new(var)
    spec$add_step(stp)
  }

  spec
}

#' Creates a categorical column specification
#'
#' @inheritParams step_numeric_column
#' @param vocabulary_list An ordered iterable defining the vocabulary. Each
#'   feature is mapped to the index of its value (if present) in vocabulary_list.
#'   Must be castable to `dtype`. If `NULL` the vocabulary will be defined as
#'   all unique values in the dataset provided when fitting the specification.
#' @param dtype The type of features. Only string and integer types are supported.
#'   If `NULL`, it will be inferred from `vocabulary_list`.
#' @param default_value The integer ID value to return for out-of-vocabulary feature
#'   values, defaults to `-1`. This can not be specified with a positive
#'   num_oov_buckets.
#' @param num_oov_buckets Non-negative integer, the number of out-of-vocabulary buckets.
#'   All out-of-vocabulary inputs will be assigned IDs in the range
#'   `[lenght(vocabulary_list), length(vocabulary_list)+num_oov_buckets)` based on a hash of
#'   the input value. A positive num_oov_buckets can not be specified with
#'   default_value.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ thal) %>%
#'   step_categorical_column_with_vocabulary_list(thal)
#'
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
#' }
#'
#' @return a `FeatureSpec` object.
#' @seealso [steps] for a complete list of allowed steps.
#'
#' @family Feature Spec Functions
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

#' Creates a categorical column with hash buckets specification
#'
#' Represents sparse feature where ids are set by hashing.
#'
#' @inheritParams step_numeric_column
#' @param hash_bucket_size An int > 1. The number of buckets.
#' @param dtype The type of features. Only string and integer types are supported.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ thal) %>%
#'   step_categorical_column_with_hash_bucket(thal, hash_bucket_size = 3)
#'
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
#' }
#'
#' @return a `FeatureSpec` object.
#' @seealso [steps] for a complete list of allowed steps.
#'
#' @family Feature Spec Functions
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

#' Create a categorical column with identity
#'
#' Use this when your inputs are integers in the range `[0-num_buckets)`.
#'
#' @inheritParams step_numeric_column
#' @param num_buckets Range of inputs and outputs is `[0, num_buckets)`.
#' @param default_value If `NULL`, this column's graph operations will fail
#'   for out-of-range inputs. Otherwise, this value must be in the range
#'   `[0, num_buckets)`, and will replace inputs in that range.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#'
#' hearts$thal <- as.integer(as.factor(hearts$thal)) - 1L
#'
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ thal) %>%
#'   step_categorical_column_with_identity(thal, num_buckets = 5)
#'
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
#' }
#'
#' @return a `FeatureSpec` object.
#' @seealso [steps] for a complete list of allowed steps.
#'
#' @family Feature Spec Functions
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

#' Creates a categorical column with vocabulary file
#'
#' Use this function when the vocabulary of a categorical variable
#' is written to a file.
#'
#' @inheritParams step_numeric_column
#' @param vocabulary_file The vocabulary file name.
#' @param vocabulary_size Number of the elements in the vocabulary. This
#'   must be no greater than length of `vocabulary_file`, if less than
#'   length, later values are ignored. If None, it is set to the length of
#'   `vocabulary_file`.
#' @param dtype The type of features. Only string and integer types are
#'   supported.
#' @param default_value The integer ID value to return for out-of-vocabulary
#'   feature values, defaults to `-1`. This can not be specified with a
#'   positive `num_oov_buckets`.
#' @param num_oov_buckets Non-negative integer, the number of out-of-vocabulary
#'   buckets. All out-of-vocabulary inputs will be assigned IDs in the range
#'   `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
#'   the input value. A positive `num_oov_buckets` can not be specified with
#'   default_value.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' file <- tempfile()
#' writeLines(unique(hearts$thal), file)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ thal) %>%
#'   step_categorical_column_with_vocabulary_file(thal, vocabulary_file = file)
#'
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
#' }
#'
#' @return a `FeatureSpec` object.
#' @seealso [steps] for a complete list of allowed steps.
#'
#' @family Feature Spec Functions
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

#' Creates Indicator Columns
#'
#' Use this step to create indicator columns from categorical columns.
#'
#' @inheritParams step_numeric_column
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' file <- tempfile()
#' writeLines(unique(hearts$thal), file)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ thal) %>%
#'   step_categorical_column_with_vocabulary_list(thal) %>%
#'   step_indicator_column(thal)
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
#' }
#' @return a `FeatureSpec` object.
#' @seealso [steps] for a complete list of allowed steps.
#'
#' @family Feature Spec Functions
#' @export
step_indicator_column <- function(spec, ...) {
  step_(spec, ..., step = StepIndicatorColumn$new, args = list(), prefix = "indicator")
}

#' Creates embeddings columns
#'
#' Use this step to create ambeddings columns from categorical
#' columns.
#'
#' @inheritParams step_numeric_column
#' @param dimension An integer specifying dimension of the embedding, must be > 0.
#'   Can also be a function of the size of the vocabulary.
#' @param combiner A string specifying how to reduce if there are multiple entries in
#'   a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with 'mean' the
#'   default. 'sqrtn' often achieves good accuracy, in particular with bag-of-words
#'   columns. Each of this can be thought as example level normalizations on
#'   the column. For more information, see `tf.embedding_lookup_sparse`.
#' @param initializer A variable initializer function to be used in embedding
#'   variable initialization. If not specified, defaults to
#'   `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
#'   `1/sqrt(dimension)`.
#' @param ckpt_to_load_from String representing checkpoint name/pattern from
#'   which to restore column weights. Required if `tensor_name_in_ckpt` is
#'   not `NULL`.
#' @param tensor_name_in_ckpt Name of the Tensor in ckpt_to_load_from from which to
#'   restore the column weights. Required if `ckpt_to_load_from` is not `NULL`.
#' @param max_norm If not `NULL`, embedding values are l2-normalized to this value.
#' @param trainable Whether or not the embedding is trainable. Default is `TRUE`.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' file <- tempfile()
#' writeLines(unique(hearts$thal), file)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ thal) %>%
#'   step_categorical_column_with_vocabulary_list(thal) %>%
#'   step_embedding_column(thal, dimension = 3)
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
#' }
#' @return a `FeatureSpec` object.
#' @seealso [steps] for a complete list of allowed steps.
#'
#' @family Feature Spec Functions
#' @export
step_embedding_column <- function(spec, ..., dimension = function(x) {as.integer(x^0.25)},
                                  combiner = "mean",
                                  initializer = NULL, ckpt_to_load_from = NULL,
                                  tensor_name_in_ckpt = NULL, max_norm = NULL,
                                  trainable = TRUE) {


  if (is.numeric(dimension))
    dimension_ <- as.integer(dimension)
  else if (rlang::is_function(dimension))
    dimension_ <- function(x) {as.integer(dimension(x))}

  args <- list(
    dimension = dimension_,
    combiner = combiner,
    initializer = initializer,
    ckpt_to_load_from = ckpt_to_load_from,
    tensor_name_in_ckpt = tensor_name_in_ckpt,
    max_norm = max_norm,
    trainable = trainable
  )

  step_(spec, ..., step = StepEmbeddingColumn$new, args = args, prefix = "embedding")
}

#' Creates bucketized columns
#'
#' Use this step to create bucketized columns from numeric columns.
#'
#' @inheritParams step_numeric_column
#' @param boundaries A sorted list or tuple of floats specifying the boundaries.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' file <- tempfile()
#' writeLines(unique(hearts$thal), file)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ age) %>%
#'   step_numeric_column(age) %>%
#'   step_bucketized_column(age, boundaries = c(10, 20, 30))
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
#' }
#' @return a `FeatureSpec` object.
#' @seealso [steps] for a complete list of allowed steps.
#'
#' @family Feature Spec Functions
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

#' Creates crosses of categorical columns
#'
#' Use this step to create crosses between categorical columns.
#'
#' @inheritParams step_numeric_column
#' @param hash_bucket_size An int > 1. The number of buckets.
#' @param hash_key (optional) Specify the hash_key that will be used by the
#'   FingerprintCat64 function to combine the crosses fingerprints on
#'   SparseCrossOp.
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' file <- tempfile()
#' writeLines(unique(hearts$thal), file)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ age) %>%
#'   step_numeric_column(age) %>%
#'   step_bucketized_column(age, boundaries = c(10, 20, 30))
#' spec_fit <- fit(spec)
#' final_dataset <- hearts %>% dataset_use_spec(spec_fit)
#' }
#' @return a `FeatureSpec` object.
#' @seealso [steps] for a complete list of allowed steps.
#'
#' @family Feature Spec Functions
#' @export
step_crossed_column <- function(spec, ..., hash_bucket_size, hash_key = NULL) {

  args <- list(
    hash_bucket_size = as.integer(hash_bucket_size),
    hash_key = hash_key
  )

  step_multiple_(spec, ..., step = StepCrossedColumn$new, args = args, prefix = "crossed")
}

#' Creates shared embeddings for categorical columns
#'
#' This is similar to [step_embedding_column], except that it produces a list of
#' embedding columns that share the same embedding weights.
#'
#' @inheritParams step_embedding_column
#' @param shared_embedding_collection_name Optional collective name of
#'   these columns. If not given, a reasonable name will be chosen based on
#'   the names of categorical_columns.
#'
#' @note Does not work in the eager mode.
#'
#' @return a `FeatureSpec` object.
#' @seealso [steps] for a complete list of allowed steps.
#'
#' @family Feature Spec Functions
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

# Input from spec ---------------------------------------------------------

#' Creates a list of inputs from a dataset
#'
#' Create a list ok Keras input layers that can be used together
#' with [keras::layer_dense_features()].
#'
#' @param dataset a TensorFlow dataset or a data.frame
#' @return a list of Keras input layers
#'
#' @examples
#' \dontrun{
#' library(tfdatasets)
#' data(hearts)
#' hearts <- tensor_slices_dataset(hearts) %>% dataset_batch(32)
#'
#' # use the formula interface
#' spec <- feature_spec(hearts, target ~ age + slope) %>%
#'   step_numeric_column(age, slope) %>%
#'   step_bucketized_column(age, boundaries = c(10, 20, 30))
#'
#' spec <- fit(spec)
#' dataset <- hearts %>% dataset_use_spec(spec)
#'
#' input <- layer_input_from_dataset(dataset)
#' }
#'
#' @export
layer_input_from_dataset <- function(dataset) {

  # only needs the head to infer types, colnames and etc.
  if (inherits(dataset, "data.frame") || inherits(dataset, "list"))
    dataset <- tensor_slices_dataset(utils::head(dataset))

  dataset <- dataset_map(dataset, ~.x)

  col_names <- column_names(dataset)
  col_types <- output_types(dataset)
  col_shapes <- output_shapes(dataset)

  inputs <- list()
  for (i in seq_along(col_names)) {

    x <- list(keras::layer_input(
      name = col_names[i],
      shape = col_shapes[[i]]$as_list()[-1],
      dtype = col_types[[i]]$name
    ))
    names(x) <- col_names[i]
    inputs <- append(inputs, x)

  }

  reticulate::dict(inputs)
}


#' Dense Features
#'
#' Retrives the Dense Features from a spec.
#'
#' @inheritParams step_numeric_column
#'
#' @return A list of feature columns.
#'
#' @export
dense_features <- function(spec) {
  spec$dense_features()
}
