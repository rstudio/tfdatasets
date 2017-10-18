


#' A dataset comprising records from one or more TFRecord files.
#'
#' @inheritParams text_line_dataset
#'
#' @export
tfrecord_dataset <- function(filenames, compression_type = "auto") {

  # validate during dataset construction
  validate_tf_version()

  # resolve filenames
  filenames <- resolve_filenames(filenames)

  # determine compression type
  if (identical(compression_type, "auto"))
    compression_type <- auto_compression_type(filenames)

  as_tf_dataset(
    tf$data$TFRecordDataset(filenames, compression_type = compression_type)
  )

}
