

#' A dataset of fixed-length records from one or more binary files.
#'
#' @param filenames A string tensor containing one or more filenames.
#' @param record_bytes An integer representing the number of bytes in each
#'   record.
#' @param header_bytes (Optional) An integer scalar representing the number of
#'   bytes to skip at the start of a file.
#' @param footer_bytes (Optional) A integer scalar representing the number of
#'   bytes to ignore at the end of a file.
#' @param buffer_size (Optional) A integer scalar representing the number of
#'   bytes to buffer when reading.
#'
#' @return A dataset
#'
#' @export
fixed_length_record_dataset <- function(filenames, record_bytes,
                                        header_bytes = NULL, footer_bytes = NULL,
                                        buffer_size = NULL) {
  as_tf_dataset(
    tf$data$FixedLengthRecordDataset(
      filenames = filenames,
      record_bytes = as_integer_tensor(record_bytes),
      header_bytes = as_integer_tensor(header_bytes),
      footer_bytes = as_integer_tensor(footer_bytes),
      buffer_size = as_integer_tensor(buffer_size)
    )
  )
}
