

#' Create a dataset comprising lines from one or more text files.
#'
#' @param filenames Characater vector containing one or more filenames.
#' @param compression_type A string, one of: `""` (no compression), `"ZLIB"`, or `"GZIP"`.
#'
#' @template roxlate-create-dataset
#'
#' @return A dataset
#'
#' @family text datasets
#'
#' @export
text_line_dataset <- function(filenames, compression_type = NULL, shuffle = NULL, batch_size = NULL, repeated = NULL) {

  dataset <- tf$contrib$data$TextLineDataset(filenames, compression_type = compression_type)

  if (!missing(shuffle))
    dataset <- dataset_shuffle(dataset, buffer_size = shuffle)
  if (!missing(repeated))
    dataset <- dataset_repeat(dataset, count = repeated)
  if (!missing(batch_size))
    dataset <- dataset_batch(dataset, batch_size)

  dataset

}





