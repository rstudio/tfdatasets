

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


#' Create a dataset from a text file with comma separated values
#'
#' @inheritParams text_line_dataset
#'
#' @param col_names Either `TRUE`, `FALSE` or a character vector of column
#'   names.
#'
#'   If `TRUE`, the first row of the input will be used as the column names, and
#'   will not be included in dataset. If `FALSE`, column names will be generated
#'   automatically: X1, X2, X3 etc.
#'
#'   If `col_names` is a character vector, the values will be used as the names
#'   of the columns, and the first row of the input will be read into the first
#'   row of the datset.
#'
#' @param col_defaults List of default values for columns. Default values must
#'   be of type integer, numeric, or character. Used both to indicate the type
#'   of each field as well as to provide defaults for missing values.
#'
#' @param field_delim An optional string. Defaults to ",". char delimiter to
#'   separate fields in a record.
#'
#' @param skip Number of lines to skip before reading data. Note that if
#'   `col_names` is explicitly provided and there are column names witin the CSV
#'   file then `skip` should be set to 1 to ensure that the column names are
#'   bypassed.
#'
#' @param num_threads (Optional) An integer representing the number of threads
#'   to use for processing elements in parallel. If not specified, elements will
#'   be processed sequentially without buffering.
#'
#' @param output_buffer_size (Optional) An integer representing the maximum
#'   number of processed elements that will be buffered when processing in
#'   parallel.
#'
#' @section Column Names:
#'
#'   Column names are not an intrinsic property of TensorFlow datasets, however
#'   they are supported in this interface to facilitate specifying features and
#'   response variables when creating input functions and generators that draw
#'   from the dataset.
#'
#' @importFrom utils read.csv
#'
#' @export
csv_dataset <- function(filenames, compression_type = NULL,
                        col_names = TRUE, col_defaults = NULL,
                        field_delim = ",", skip = 0,
                        num_threads = NULL, output_buffer_size = NULL) {

  # read the first 1000 rows to faciliate deduction of column names / types as well
  # as checking that any specified col_names or col_defaults have the correct length
  preview <- text_line_dataset(filenames[[1]], compression_type) %>%
    dataset_skip(skip) %>%
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
    header = isTRUE(col_names),
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
  if (isTRUE(col_names)) {
    col_names <- names(preview_csv)
    skip <- skip + 1
  } else if (is.character(col_names)) {
    validate_columns(col_names, 'col_names')
  } else {
    col_names <- paste0("X", 1:ncol(preview_csv))
  }

  # resolve/validate record defaults
  if (!is.null(col_defaults)) {
    validate_columns(col_defaults, 'col_defaults')
    record_defaults <- lapply(col_defaults, function(x) {
      list(x)
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
  dataset <- text_line_dataset(filenames, compression_type) %>%
    dataset_skip(skip) %>%
    dataset_map(
      map_func = function(line) {
        tf$decode_csv(
          records = line,
          record_defaults = record_defaults,
          field_delim = field_delim
        )
      },
      num_threads = num_threads,
      output_buffer_size = output_buffer_size
    )

  # set the col_names on the dataset (used in e.g. input_fn_from_dataset)
  dataset$`_col_names` <- col_names

  # return the dataset
  dataset
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






