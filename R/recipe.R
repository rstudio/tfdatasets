Recipe <- R6::R6Class(
  "Recipe",
  public = list(
    base_steps = list(),
    steps = list(),
    formula = NULL,
    column_names = NULL,
    column_types = NULL,
    dataset = NULL,

    initialize = function(formula, dataset) {
      self$formula <- formula
      self$dataset <- dataset

      self$column_names <- column_names(dataset)
      self$column_types <- output_types(dataset)
    },

    add_step = function(step) {
      if (inherits(step, "StepNumericColumn") ||
        inherits(step, "StepCategoricalColumnWithVocabularyList")) {
        self$base_steps <- append(self$base_steps, step)
      }

      if (!inherits(step, "StepCategoricalColumnWithVocabularyList")) {
        self$steps <- append(self$steps, step)
      }
    },

    fit = function() {
      ds <- reticulate::as_iterator(self$dataset)
      nxt <- reticulate::iter_next(ds)

      while (!is.null(nxt)) {
        for (i in seq_along(self$base_steps)) {
          self$base_steps[[i]]$fit_batch(nxt)
        }
        nxt <- reticulate::iter_next(ds)
      }

      for (i in seq_along(self$base_steps)) {
        self$base_steps[[i]]$fit_resume()
      }
    },

    base_features = function() {
      feats <- lapply(self$base_steps, function(x) x$feature())
      names(feats) <- sapply(self$base_steps, function(x) x$key)

      feats
    },

    features = function() {
      feats <- lapply(self$steps, function(x) {
        print(x$feature)
        x$feature()
      })
      feats
    }
  ),

  private = list(
    deep_clone = function(name, value) {
      if (inherits(value, "R6")) {
        value$clone(deep = TRUE)
      } else if (name == "steps" || name == "base_steps") {
        lapply(value, function(x) x$clone(deep = TRUE))
      } else {
        value
      }
    }
  )
)

Step <- R6::R6Class(
  classname = "Step",
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

StepNumericColumn <- R6::R6Class(
  "StepNumericColumn",
  inherit = Step,
  public = list(
    key = NULL,
    shape = NULL,
    default_value = NULL,
    dtype = NULL,
    normalizer_fn = NULL,
    initialize = function(key, shape, default_value, dtype, normalizer_fn) {
      self$key <- key
      self$shape <- shape
      self$default_value <- default_value
      self$dtype <- dtype
      self$normalizer_fn <- normalizer_fn
    },
    fit_batch = function(batch) {
      if (is.null(self$normalizer_fn) || is.function(self$normalizer_fn)) {

      } else {

      }
    },
    fit_resume = function() {

    },
    feature = function () {
      tf$feature_column$numeric_column(
        key = self$key, shape = self$shape,
        default_value = self$default_value,
        dtype = self$dtype,
        normalizer_fn = self$normalizer_fn
      )
    }
  )
)

StepCategoricalColumnWithVocabularyList <- R6::R6Class(
  "StepCategoricalColumnWithVocabularyList",
  inherit = Step,
  public = list(

    key = NULL,
    vocabulary_list = NULL,
    dtype = NULL,
    default_value = -1L,
    num_oov_buckets = 0L,
    vocabulary_list_aux = NULL,

    initialize = function(key, vocabulary_list = NULL, dtype = NULL, default_value = -1L,
                          num_oov_buckets = 0L) {

      self$key <- key
      self$vocabulary_list <- vocabulary_list
      self$dtype = dtype
      self$default_value <- default_value
      self$num_oov_buckets <- num_oov_buckets

    },

    fit_batch = function(batch) {

      if (is.null(self$vocabulary_list)) {
        values <- batch[[self$key]]
        unq <- unique(values$numpy())
        self$vocabulary_list_aux <- sort(unique(c(self$vocabulary_list_aux, unq)))
      }

    },

    fit_resume = function() {

      if (is.null(self$vocabulary_list)) {
        self$vocabulary_list <- self$vocabulary_list_aux
      }

    },

    feature = function() {

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

StepIndicatorColumn <- R6::R6Class(
  "StepIndicatorColumn",
  inherit = Step,
  public = list(
    categorical_column = NULL,
    base_features = NULL,
    initialize = function(categorical_column, base_features) {
      self$categorical_column = categorical_column
      self$base_features <- base_features
    },
    feature = function() {
      base_features <- self$base_features()
      tf$feature_column$indicator_column(base_features[[self$categorical_column]])
    }
  )
)

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
    base_features = NULL,

    initialize = function(categorical_column, dimension, combiner = "mean", initializer = NULL,
                          ckpt_to_load_from = NULL, tensor_name_in_ckpt = NULL, max_norm = NULL,
                          trainable = TRUE, base_features) {

      self$categorical_column <- categorical_column
      self$dimension <- dimension
      self$combiner <- combiner
      self$initializer <- initializer
      self$ckpt_to_load_from <- ckpt_to_load_from
      self$tensor_name_in_ckpt <- tensor_name_in_ckpt
      self$max_norm <- max_norm
      self$trainable <- trainable
      self$base_features <- base_features

    },

    feature = function() {

      base_features <- self$base_features()
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

recipe <- function(formula, dataset) {
  rec <- Recipe$new(formula, dataset)
  rec
}

step_numeric_column <- function(rec, ..., shape = 1L, default_value = NULL,
                                dtype = tf$float32, normalizer_fn = NULL) {

  rec <- rec$clone(deep = TRUE)
  variables <- tidyselect::vars_select(rec$column_names, !!!quos(...))
  for (var in variables) {
    stp <- StepNumericColumn$new(var, shape, default_value, dtype, normalizer_fn)
    rec$add_step(stp)
  }

  rec
}

step_categorical_column_with_vocabulary_list <- function(rec, ..., vocabulary_list = NULL,
                                                         dtype = NULL, default_value = -1L,
                                                         num_oov_buckets = 0L) {
  rec <- rec$clone(deep = TRUE)
  variables <- tidyselect::vars_select(rec$column_names, !!!quos(...))
  for (var in variables) {
    stp <- StepCategoricalColumnWithVocabularyList$new(
      var, vocabulary_list, dtype,
      default_value, num_oov_buckets
    )
    rec$add_step(stp)
  }

  rec
}

step_indicator_column <- function(rec, ...) {

  rec <- rec$clone(deep = TRUE)
  variables <- tidyselect::vars_select(rec$column_names, !!!quos(...))
  for (var in variables) {
    stp <- StepIndicatorColumn$new(var, rec$base_features)
    rec$add_step(stp)
  }

  rec
}

step_embedding_column <- function(rec, ..., dimension, combiner = "mean",
                                  initializer = NULL, ckpt_to_load_from = NULL,
                                  tensor_name_in_ckpt = NULL, max_norm = NULL,
                                  trainable = TRUE) {
  rec <- rec$clone(deep = TRUE)
  variables <- tidyselect::vars_select(rec$column_names, !!!quos(...))
  for (var in variables) {
    stp <- StepEmbeddingColumn$new(var, dimension, combiner, initializer,
                                   ckpt_to_load_from, tensor_name_in_ckpt,
                                   max_norm, trainable,
                                   base_features = rec$base_features)
    rec$add_step(stp)
  }

  rec
}



