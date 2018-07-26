

#' Samples elements at random from the datasets in `datasets`.
#'
#' @param datasets A list ofobjects with compatible structure.
#' @param weights (Optional.) A list of `length(datasets)` floating-point values where
#'   `weights[[i]]` represents the probability with which an element should be sampled
#'   from `datasets[[i]]`, or a dataset object where each element is such a list.
#'   Defaults to a uniform distribution across `datasets`.
#' @param seed (Optional.) An integer, representing the random seed
#'   that will be used to create the distribution.
#'
#' @return A dataset that interleaves elements from `datasets` at random, according to
#'   `weights` if provided, otherwise with uniform probability.
#'
#' @export
sample_from_datasets <- function(datasets, weights = NULL, seed = NULL) {
  validate_tf_version()
  as_tf_dataset(
    tf$contrib$data$sample_from_datasets(
      datasets,
      weights = weights,
      seed = as_integer_tensor(seed))
  )
}



