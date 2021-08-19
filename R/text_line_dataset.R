

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
        decoded <- tfio_decode_csv(
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


#' Reads CSV files into a batched dataset
#'
#' Reads CSV files into a dataset, where each element is a (features, labels) list that
#' corresponds to a batch of CSV rows. The features dictionary maps feature column names
#' to tensors containing the corresponding feature data, and labels is a tensor
#' containing the batch's label data.
#'
#' @param file_pattern List of files or glob patterns of file paths containing CSV records.
#' @param batch_size An integer representing the number of records to combine in a single
#'   batch.
#' @param column_names An optional list of strings that corresponds to the CSV columns, in
#'   order. One per column of the input record. If this is not provided, infers the column
#'   names from the first row of the records. These names will be the keys of the features
#'   dict of each dataset element.
#' @param column_defaults A optional list of default values for the CSV fields. One item
#'   per selected column of the input record. Each item in the list is either a valid CSV
#'   dtype (integer, numeric, or string), or a tensor with one of the
#'   aforementioned types. The tensor can either be a scalar default value (if the column
#'   is optional), or an empty tensor (if the column is required). If a dtype is provided
#'   instead of a tensor, the column is also treated as required. If this list is not
#'   provided, tries to infer types based on reading the first `num_rows_for_inference` rows
#'   of files specified, and assumes all columns are optional, defaulting to `0` for
#'   numeric values and `""` for string values. If both this and `select_columns` are
#'   specified, these must have the same lengths, and `column_defaults` is assumed to be
#'   sorted in order of increasing column index.
#' @param label_name A optional string corresponding to the label column. If provided, the
#'   data for this column is returned as a separate tensor from the features dictionary,
#'   so that the dataset complies with the format expected by a TF Estiamtors and Keras.
#' @param select_columns (Ignored if using TensorFlow version 1.8.) An optional list of
#'   integer indices or string column names, that specifies a subset of columns of CSV data
#'   to select. If column names are provided, these must correspond to names provided in
#'   `column_names` or inferred from the file header lines. When this argument is specified,
#'   only a subset of CSV columns will be parsed and returned, corresponding to the columns
#'   specified. Using this results in faster parsing and lower memory usage. If both this
#'   and `column_defaults` are specified, these must have the same lengths, and
#'   `column_defaults` is assumed to be sorted in order of increasing column index.
#' @param field_delim An optional string. Defaults to `","`. Char delimiter to separate
#'   fields in a record.
#' @param use_quote_delim An optional bool. Defaults to `TRUE`. If false, treats double
#'   quotation marks as regular characters inside of the string fields.
#' @param na_value Additional string to recognize as NA/NaN.
#' @param header A bool that indicates whether the first rows of provided CSV files
#'   correspond to header lines with column names, and should not be included in the data.
#' @param num_epochs An integer specifying the number of times this dataset is repeated. If
#'   NULL, cycles through the dataset forever.
#' @param shuffle A bool that indicates whether the input should be shuffled.
#' @param shuffle_buffer_size Buffer size to use for shuffling. A large buffer size
#'   ensures better shuffling, but increases memory usage and startup time.
#' @param shuffle_seed Randomization seed to use for shuffling.
#' @param prefetch_buffer_size An int specifying the number of feature batches to prefetch
#'   for performance improvement. Recommended value is the number of batches consumed per
#'   training step.
#' @param num_parallel_reads Number of threads used to read CSV records from files. If >1,
#'   the results will be interleaved.
#' @param num_parallel_parser_calls (Ignored if using TensorFlow version 1.11 or later.)
#'   Number of parallel invocations of the CSV parsing function on CSV records.
#' @param sloppy If `TRUE`, reading performance will be improved at the cost of
#'   non-deterministic ordering. If `FALSE`, the order of elements produced is
#'   deterministic prior to shuffling (elements are still randomized if `shuffle=TRUE`.
#'   Note that if the seed is set, then order of elements after shuffling is
#'   deterministic). Defaults to `FALSE`.
#' @param num_rows_for_inference Number of rows of a file to use for type inference if
#'   record_defaults is not provided. If `NULL`, reads all the rows of all the files.
#'   Defaults to 100.
#'
#' @return A dataset, where each element is a (features, labels) list that corresponds to
#'   a batch of `batch_size` CSV rows. The features dictionary maps feature column names
#'   to tensors containing the corresponding column data, and labels is a tensor
#'   containing the column data for the label column specified by `label_name`.
#'
#' @export
make_csv_dataset <- function(file_pattern, batch_size,
                             column_names = NULL, column_defaults = NULL,
                             label_name = NULL, select_columns = NULL,
                             field_delim = ",", use_quote_delim = TRUE, na_value = "",
                             header = TRUE, num_epochs = NULL,
                             shuffle = TRUE, shuffle_buffer_size = 10000, shuffle_seed = NULL,
                             prefetch_buffer_size = 1,
                             num_parallel_reads = 1, num_parallel_parser_calls = 2,
                             sloppy = FALSE, num_rows_for_inference = 100) {
  ## valdiate version
  validate_tf_version("1.8", "make_csv_dataset")

  ## Gather arguments common to all tensorflow versions
  args <- list(
    file_pattern = file_pattern,
    batch_size = as.integer(batch_size),
    column_names = column_names,
    column_defaults = column_defaults,
    label_name = label_name,
    field_delim = field_delim,
    use_quote_delim = use_quote_delim,
    na_value = na_value,
    header = header,
    num_epochs = num_epochs,
    shuffle = shuffle,
    shuffle_buffer_size = as.integer(shuffle_buffer_size),
    shuffle_seed = shuffle_seed,
    prefetch_buffer_size = as.integer(prefetch_buffer_size),
    num_parallel_reads = as.integer(num_parallel_reads),
    sloppy = sloppy,
    num_rows_for_inference = as.integer(num_rows_for_inference)
  )

  tf_ver = tensorflow::tf_version()

  ## TensorFlow version 1.8 rejects "select_columns"
  if (tf_ver > "1.8") {
    args[['select_columns']] <- select_columns
  }

  ## TensorFlow versions prior to 1.11 accept "num_parallel_parser_calls"
  if (tf_ver < "1.11") {
    args[['num_parallel_parser_calls']] <- as.integer(num_parallel_parser_calls)
  }

  do.call(tfd_make_csv_dataset, args)
}
