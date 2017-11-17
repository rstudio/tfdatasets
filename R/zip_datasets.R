


#' Creates a dataset by zipping together the given datasets.
#'
#' Merges datasets together into pairs or tuples that contain an element from
#' each dataset.
#'
#' @param ... Datasets to zip (or a single argument with a list or list of lists of datasets).
#'
#' @return A dataset
#'
#' @export
zip_datasets <- function(...) {
  as_tf_dataset(
    tf_data$Dataset$zip(tuple(...))
  )
}

