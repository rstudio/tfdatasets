

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
#' @inheritParams choose_from_datasets
#'
#' @return A dataset that interleaves elements from `datasets` at random, according to
#'   `weights` if provided, otherwise with uniform probability.
#'
#' @export
sample_from_datasets <-
function(datasets, weights = NULL, seed = NULL, stop_on_empty_dataset = TRUE)
{
  args <- capture_args(list(seed = as_integer_tensor),
                       ignore = "dataset")
  validate_tf_version()
  if (tf_version() >= "2.7") {
    callable <- tf$data$Dataset$sample_from_datasets
  } else {
    callable <- tf$data$experimental$sample_from_datasets
  }
  as_tf_dataset(do.call(callable, args))
}


#' Creates a dataset that deterministically chooses elements from datasets.
#'
#' @param datasets A non-empty list of tf.data.Dataset objects with compatible
#'   structure.
#' @param choice_dataset A `tf.data.Dataset` of scalar `tf.int64` tensors
#'   between `0` and `length(datasets) - 1`.
#' @param stop_on_empty_dataset If `TRUE`, selection stops if it encounters an
#'   empty dataset. If `FALSE`, it skips empty datasets. It is recommended to
#'   set it to `TRUE`. Otherwise, the selected elements start off as the user
#'   intends, but may change as input datasets become empty. This can be
#'   difficult to detect since the dataset starts off looking correct. Defaults
#'   to `TRUE`.
#'
#' @return Returns a dataset that interleaves elements from datasets according
#'   to the values of choice_dataset.
#' @export
#'
#' @examples
#' \dontrun{
#' datasets <- list(tensors_dataset("foo") %>% dataset_repeat(),
#'                  tensors_dataset("bar") %>% dataset_repeat(),
#'                  tensors_dataset("baz") %>% dataset_repeat())
#'
#' # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.
#' choice_dataset <- range_dataset(0, 3) %>% dataset_repeat(3)
#' result <- choose_from_datasets(datasets, choice_dataset)
#' result %>% as_array_iterator() %>% iterate(function(s) s$decode()) %>% print()
#' # [1] "foo" "bar" "baz" "foo" "bar" "baz" "foo" "bar" "baz"
#' }
choose_from_datasets <- function(datasets, choice_dataset, stop_on_empty_dataset=TRUE) {
  require_tf_version("2.7", "choose_from_datasets")
  tf$data$Dataset$choose_from_datasets(datasets, choice_dataset, stop_on_empty_dataset)
}
