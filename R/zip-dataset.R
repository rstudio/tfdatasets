


#' Creates a dataset by zipping together the given datasets.
#'
#' Merges datasets together into pairs or tuples that contain an element from
#' each dataset.
#'
#' @param datasets A nested structure (e.g. list or list of lists) of datasets.
#'
#' @return A dataset
#'
#' @export
zip_datasets <- function(datasets) {
  tf$contrib$data$Dataset$zip(tuple(datasets))
}

