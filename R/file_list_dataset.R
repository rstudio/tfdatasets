


#' A dataset of all files matching a pattern
#'
#' @param file_pattern A string, representing the filename pattern that will be matched.
#' @param shuffle  (Optional) If `TRUE``, the file names will be shuffled randomly.
#'   Defaults to `TRUE`
#' @param seed  (Optional) An integer, representing the random seed that
#'   will be used to create the distribution.
#'
#' @return A dataset of string correponding to file names
#'
#' @note The `shuffle` and `seed` arguments only apply for TensorFlow >= v1.8
#'
#' @details
#'
#' For example, if we had the following files on our filesystem: - /path/to/dir/a.txt -
#' /path/to/dir/b.csv - /path/to/dir/c.csv
#'
#' If we pass "/path/to/dir/*.csv" as the `file_pattern`, the dataset would produce: -
#' /path/to/dir/b.csv - /path/to/dir/c.csv
#'
#' @export
file_list_dataset <- function(file_pattern, shuffle = NULL, seed = NULL) {

  # validate during dataset contruction
  validate_tf_version()

  args <- list(file_pattern = file_pattern)

  # validate params
  if (!missing(shuffle)) {
    validate_tf_version("1.8", "shuffle")
    args$shuffle <- shuffle
  }
  if (!missing(seed)) {
    validate_tf_version("1.8", "seed")
    args$seed <- as_integer_tensor(seed)
  }

  # create dataset
  as_tf_dataset(
    do.call(tf$data$Dataset$list_files, args)
  )
}
