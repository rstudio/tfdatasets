


#' A dataset comprising records from one or more TFRecord files.
#'
#' @inheritParams text_line_dataset
#'
#' @export
tfrecord_dataset <- function(filenames, compression_type = NULL) {

  # validate during dataset construction
  validate_tf_version()

  # resolve NULL compression
  if (is.null(compression_type))
    compression_type <- ""

  as_tf_dataset(
    tf$data$TFRecordDataset(filenames, compression_type = compression_type)
  )

}
