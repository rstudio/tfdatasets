

#' A dataset comprising lines from one or more text files.
#'
#' @param filenames Characater vector containing one or more filenames or
#'   filename glob patterns (e.g. "train.csv", "*.csv", "*-train.csv", etc.)
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

  # resolve filename wildcards
  filenames <- resolve_filenames(filenames)

  # determine compression type
  if (identical(compression_type, "auto"))
    compression_type <- auto_compression_type(filenames)

  tf$contrib$data$TextLineDataset(filenames, compression_type = compression_type)
}


#' Create a dataset from a text file with comma separated values
#'
#' @inheritParams text_line_dataset
#' @inheritParams dataset_decode_csv
#'
#' @param skip Number of lines to skip before reading data. Note that if
#'   `col_names` is explicitly provided and there are column names witin the CSV
#'   file then `skip` should be set to 1 to ensure that the column names are
#'   bypassed.
#'
#' @note The [csv_dataset()] function is a convenience wrapper for the
#'   [text_line_dataset()] and [dataset_decode_csv()] functions.
#'
#' @export
csv_dataset <- function(filenames, compression_type = NULL,
                        col_names = NULL, record_defaults = NULL,
                        field_delim = ",", skip = 0,
                        num_threads = NULL, output_buffer_size = NULL) {
  text_line_dataset(filenames, compression_type = compression_type) %>%
    dataset_skip(skip) %>%
    dataset_decode_csv(
      col_names = col_names,
      record_defaults = record_defaults,
      field_delim = field_delim,
      num_threads = num_threads,
      output_buffer_size = output_buffer_size
    )
}


#' Transform a dataset with CSV text lines into a dataset with named columns
#'
#' @param dataset Dataset with CSV text lines
#'
#' @param col_names Character vector with column names (or `NULL` to automatically
#'   detect the column names from the first row of the input file).
#'
#'   If `col_names` is a character vector, the values will be used as the names
#'   of the columns, and the first row of the input will be read into the first
#'   row of the datset. Note that if the underlying CSV file also includes
#'   column names in it's first row, this row should be skipped explicitly with
#'   [dataset_skip()].
#'
#'   If `NULL`, the first row of the input will be used as the column names, and
#'   will not be included in dataset.
#'
#' @param record_defaults List of default values for records. Default values
#'   must be of type integer, numeric, or character. Used both to indicate the
#'   type of each field as well as to provide defaults for missing values.
#'
#' @param field_delim An optional string. Defaults to ",". char delimiter to
#'   separate fields in a record.
#'
#' @param num_threads (Optional) An integer representing the number of threads
#'   to use for parsing csv records in parallel. If not specified, elements will
#'   be processed sequentially without buffering.
#'
#' @param output_buffer_size (Optional) An integer representing the maximum
#'   number of processed elements that will be buffered when processing in
#'   parallel.
#'
#' @importFrom utils read.csv
#'
#' @note The [csv_dataset()] function is a convenience wrapper for the
#'   [text_line_dataset()] and [dataset_decode_csv()] functions.
#'
#' @family dataset methods
#'
#' @export
dataset_decode_csv <- function(dataset, col_names = NULL, record_defaults = NULL,
                               field_delim = ",", num_threads = NULL, output_buffer_size = NULL) {

  # read the first 1000 rows to faciliate deduction of column names / types as well
  # as checking that any specified col_names or record_defaults have the correct length
  preview <- dataset %>%
    dataset_take(1000) %>%
    dataset_batch(1000)
  preview <- with_session(function(session) {
    iter <- one_shot_iterator(preview)
    session$run(iter$get_next())
  })

  # read the csv using read.csv to do column/type deduction
  preview_con <- textConnection(preview)
  on.exit(close(preview_con), add = TRUE)
  preview_csv <- read.csv(
    file = preview_con,
    header = is.null(col_names),
    sep = field_delim,
    comment.char = "",
    stringsAsFactors = FALSE
  )

  # validate columns helper
  validate_columns <- function(object, name) {
    if (length(object) != ncol(preview_csv))
      stop(sprintf(
        "Incorrect number of %s provided (dataset has %d columns)", name, ncol(preview_csv)
      ), call. = FALSE)
  }

  # resolve/validate col_names (add extra skip if we have col_names in the file)
  skip <- 0
  if (is.null(col_names)) {
    col_names <- names(preview_csv)
    skip <- 1
  } else if (is.character(col_names)) {
    validate_columns(col_names, 'col_names')
  } else {
    stop("col_names must be a character vector")
  }

  # resolve/validate record defaults
  if (!is.null(record_defaults)) {
    validate_columns(record_defaults, 'record_defaults')
    record_defaults <- lapply(record_defaults, function(x) {
      if (!is.list(x))
        list(x)
      else
        x
    })
  } else {
    record_types <- lapply(preview_csv, typeof)
    record_defaults <- lapply(unname(record_types), function(x) {
      switch(x,
             integer = list(0L),
             double = list(0),
             character = list(""),
             list("") # default
      )
    })
  }

  # read csv
  dataset %>%
    dataset_skip(skip) %>%
    dataset_map(
      map_func = function(line) {
        decoded <- tf$decode_csv(
          records = line,
          record_defaults = record_defaults,
          field_delim = field_delim
        )
        names(decoded) <- col_names
        decoded
      },
      num_threads = num_threads,
      output_buffer_size = output_buffer_size
    )
}







