

#' Create a dataset comprising lines from one or more text files.
#'
#' @param filenames Characater vector containing one or more filenames.
#' @param compression_type A string, one of: `""` (no compression), `"ZLIB"`, or `"GZIP"`.
#'
#' @return A dataset
#'
#' @family text datasets
#'
#' @export
text_line_dataset <- function(filenames, compression_type = NULL) {
  tf$contrib$data$TextLineDataset(filenames, compression_type = compression_type)
}





