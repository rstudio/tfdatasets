


#' A dataset of all files matching a pattern
#'
#' @param file_pattern A string, representing the filename pattern that will be
#'   matched.
#'
#' @return A dataset of string correponding to file names
#'
#' @details
#'
#' For example, if we had the following files on our filesystem:
#'   - /path/to/dir/a.txt
#'   - /path/to/dir/b.csv
#'   - /path/to/dir/c.csv
#'
#' If we pass "/path/to/dir/*.csv" as the `file_pattern`, the dataset would produce:
#'   - /path/to/dir/b.csv
#'   - /path/to/dir/c.csv
#'
#' @export
file_list_dataset <- function(file_pattern) {

  # validate during dataset contruction
  validate_tf_version()

  # create dataset
  as_tf_dataset(
    tf$data$Dataset$list_files(file_pattern)
  )
}

