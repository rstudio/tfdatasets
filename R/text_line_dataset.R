

#' A dataset comprising lines from one or more text files.
#'
#' @param filenames String(s) specifying one or more filenames
#' @param compression_type A string, one of: `NULL` (no compression), `"ZLIB"`, or
#'   `"GZIP"`.
#' @param record_spec (Optional) Specification used to decode delimimted text lines
#'   into records (see [delim_record_spec()]).
#'
#' @param parallel_records (Optional) An integer, representing the number of
#'   records to decode in parallel. If not specified, records will be
#'   processed sequentially.
#'
#' @return A dataset
#'
#' @family text datasets
#'
#' @export
text_line_dataset <- function(filenames, compression_type = NULL,
                              record_spec = NULL, parallel_records = NULL) {

  # validate during dataset contruction
  validate_tf_version()

  # resolve NULL to ""
  if (is.null(compression_type))
    compression_type <- ""

  # basic test line dataset
  dataset <- tf$data$TextLineDataset(
    filenames = filenames,
    compression_type = compression_type
  )

  # if a record_spec is provided then apply it
  if (!is.null(record_spec)) {
    dataset <- dataset %>%
      dataset_decode_delim(
        record_spec = record_spec,
        parallel_records = parallel_records
      )
  }

  # return dataset
  as_tf_dataset(dataset)
}



#' Transform a dataset with delimted text lines into a dataset with named
#' columns
#'
#' @param dataset Dataset containing delimited text lines (e.g. a CSV)
#'
#' @param record_spec Specification of column names and types (see [delim_record_spec()]).
#'
#' @param parallel_records (Optional) An integer, representing the number of
#'   records to decode in parallel. If not specified, records will be
#'   processed sequentially.
#'
#' @family dataset methods
#'
#' @export
dataset_decode_delim <- function(dataset, record_spec, parallel_records = NULL) {

  # read csv
  dataset <- dataset %>%
    dataset_skip(record_spec$skip) %>%
    dataset_map(
      map_func = function(line) {
        decoded <- tf$decode_csv(
          records = line,
          record_defaults = record_spec$defaults,
          field_delim = record_spec$delim
        )
        names(decoded) <- record_spec$names
        decoded
      },
      num_parallel_calls = parallel_records
    )

  # return dataset
  as_tf_dataset(dataset)
}

