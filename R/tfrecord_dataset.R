
#' A dataset comprising records from one or more TFRecord files.
#'
#' @inheritParams text_line_dataset
#' @param buffer_size An integer representing the number of bytes in the read buffer. (0
#'   means no buffering).
#' @param num_parallel_reads An integer representing the number of files to read in
#'   parallel. Defaults to reading files sequentially.
#'
#' @details If the dataset encodes a set of TFExample instances, then they can be decoded
#'   into named records using the [dataset_map()] function (see example below).
#'
#' @examples \dontrun{
#'
#' # Creates a dataset that reads all of the examples from two files, and extracts
#' # the image and label features.
#' filenames <- c("/var/data/file1.tfrecord", "/var/data/file2.tfrecord")
#' dataset <- tfrecord_dataset(filenames) %>%
#'   dataset_map(function(example_proto) {
#'     features <- list(
#'       image = tf$FixedLenFeature(shape(), tf$string, default_value = ""),
#'       label = tf$FixedLenFeature(shape(), tf$int32, default_value = 0L)
#'     )
#'     tf$parse_single_example(example_proto, features)
#'   })
#' }
#'
#' @export
tfrecord_dataset <- function(filenames,
                             compression_type = NULL,
                             buffer_size = NULL,
                             num_parallel_reads = NULL) {

  # validate during dataset construction
  validate_tf_version()

  # validate version for new parameters
  if (!missing(buffer_size))
    validate_tf_version("1.8", "buffer_size")
  if (!missing(num_parallel_reads))
    validate_tf_version("1.8", "num_parallel_reads")

  # resolve NULL compression
  if (is.null(compression_type))
    compression_type <- ""

  # read the dataset
  as_tf_dataset(
    tf$data$TFRecordDataset(filenames, compression_type = compression_type)
  )
}
