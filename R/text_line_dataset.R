

#' Create a dataset comprising lines from one or more text files.
#'
#' @param filenames Characater vector containing one or more filenames.
#' @param compression_type A string, one of: `"auto"` (determine based on file
#'   extension), `""` (no compression), `"ZLIB"`, or `"GZIP"`. For
#'   `"auto"`, GZIP will be automatically selected if any of
#'   the `filenames` have a .gz extension and ZLIB will be automatically
#'   selected if any of the `filenames` have a .zlib extension (otherwise
#'   no compression will be used).
#'
#' @return A dataset
#'
#' @family text datasets
#'
#' @export
text_line_dataset <- function(filenames, compression_type = "auto") {

  # determine compression type
  if (identical(compression_type, "auto"))
    compression_type <- auto_compression_type(filenames)

  tf$contrib$data$TextLineDataset(filenames, compression_type = compression_type)
}


auto_compression_type <- function(filenames) {
  has_ext <- function(ext) {
    any(identical(tolower(tools::file_ext(filenames)), ext))
  }
  if (has_ext("gz"))
    "GZIP"
  else if (has_ext("zlib"))
    "ZLIB"
  else
    ""
}




