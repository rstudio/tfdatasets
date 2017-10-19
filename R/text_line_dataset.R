

#' A dataset comprising lines from one or more text files.
#'
#' @param filenames String(s) specifying one or more filenames or filename
#'   patterns (e.g. "*.csv"). Alternatively, dataset with a collection of
#'   file names (e.g. like the one returned by [file_list_dataset()]).
#'
#' @param compression_type A string, one of: `NULL` (no compression), `"ZLIB"`, or
#'   `"GZIP"`.
#'
#' @param parallel_files The number of input files to read in parallel.
#'
#' @param parallel_files_block_length Records from files read in parallel will
#'   be interleaved in blocks of length `parallel_files_block_length`.
#'
#' @return A dataset
#'
#' @family text datasets
#'
#' @export
text_line_dataset <- function(filenames, compression_type = NULL,
                              parallel_files = NULL,
                              parallel_files_block_length = 1) {

  # validate during dataset contruction
  validate_tf_version()

  # filenames as dataset
  filenames <- filenames_dataset(filenames)

  # resolve NULL to ""
  if (is.null(compression_type))
    compression_type <- ""

  # define read function
  read_func <- function(file) {
    tf$data$TextLineDataset(
      filenames = file,
      compression_type = compression_type
    )
  }

  # read the dataset
  read_dataset(filenames, read_func, parallel_files, parallel_files_block_length)
}


#' Create a dataset from a text file with delimited values
#'
#' @inheritParams text_line_dataset
#' @inheritParams dataset_decode_delim
#'
#' @param skip Number of lines to skip before reading data. Note that if
#'   `col_names` is explicitly provided and there are column names witin the CSV
#'   file then `skip` should be set to 1 to ensure that the column names are
#'   bypassed.
#'
#' @note The [delim_dataset()] function is a convenience wrapper for the
#'   [text_line_dataset()] and [dataset_decode_delim()] functions.
#'
#'   The [csv_dataset()] and [tsv_dataset()] are wrappers that parse comma and
#'   tab separated text files respectively.
#'
#' @export
delim_dataset <- function(filenames, compression_type = NULL, delim, skip = 0,
                          col_names = NULL, col_types = NULL, col_defaults = NULL,
                          parallel_files = NULL,
                          parallel_files_block_length = 1,
                          parallel_records = NULL) {

  # if it's a dataset or if it is a glob pattern then we are dealing with a dataset

  # if parallel options are provided then we need to turn it into a dataset

  # if we are dealing with a dataset then we have a separate way of resolving the
  # first file name

  # if we are dealing with a dataset then we use flat_map, otherwise we don't

  # make 'filenames' much less flexible (passthrough, no auto-glob)


  # NOTE: take(-1) to enumerate all the filenames doesn't currently work as expected
  #  needs to be fixed no matter what

  # I can't do sess$run at all if I want to keep the function pure. Probably need
  # to always convert into a dataset (as we do now) then enumerate with flat_map
  # (but where is the perf issue!)

  # turn the filenames into a dataset
  filenames <- filenames_dataset(filenames)

  # if dataset options aren't specified then we need to discover them
  if (requires_preview(col_names, col_types, col_defaults)) {
    # get the first filename
    file <- with_session(function(sess) {
      sess$run(next_batch(filenames))
    })
    file_dataset <- tf$data$TextLineDataset(file, compression_type = compression_type)
    opts <- resolve_dataset_options(file_dataset, delim, col_names, col_types, col_defaults, skip)
    skip <- opts$skip
    col_names <- opts$col_names
    col_types <- opts$col_types
    col_defaults <- opts$col_defaults
  }

  # define read function
  read_func <- function(file) {
    tf$data$TextLineDataset(file, compression_type = compression_type) %>%
      dataset_skip(skip) %>%
      dataset_decode_delim(
        delim = delim,
        col_names = col_names,
        col_types = col_types,
        col_defaults = col_defaults,
        parallel_records = parallel_records
      )
  }

  # read the dataset
  read_dataset(filenames, read_func, parallel_files, parallel_files_block_length)
}


#' @rdname delim_dataset
#' @export
csv_dataset <- function(filenames, compression_type = NULL, skip = 0,
                        col_names = NULL, col_types = NULL, col_defaults = NULL,
                        parallel_files = NULL,
                        parallel_files_block_length = 1,
                        parallel_records = NULL) {
  delim_dataset(
    filenames = filenames,
    compression_type = compression_type,
    delim = ",",
    skip = skip,
    col_names = col_names,
    col_defaults = col_defaults,
    col_types = col_types,
    parallel_files = parallel_files,
    parallel_files_block_length = parallel_files_block_length,
    parallel_records = parallel_records
  )
}

#' @rdname delim_dataset
#' @export
tsv_dataset <- function(filenames, compression_type = NULL, skip = 0,
                        col_names = NULL, col_types = NULL, col_defaults = NULL,
                        parallel_files = NULL,
                        parallel_files_block_length = 1,
                        parallel_records = NULL) {
  delim_dataset(
    filenames = filenames,
    compression_type = compression_type,
    delim = "\t",
    skip = skip,
    col_names = col_names,
    col_types = col_types,
    col_defaults = col_defaults,
    parallel_files = parallel_files,
    parallel_files_block_length = parallel_files_block_length,
    parallel_records = parallel_records
  )
}


#' Transform a dataset with delimted text lines into a dataset with named
#' columns
#'
#' @param dataset Dataset containing delimited text lines (e.g. a CSV)
#'
#' @param col_names Character vector with column names (or `NULL` to
#'   automatically detect the column names from the first row of the input
#'   file).
#'
#'   If `col_names` is a character vector, the values will be used as the names
#'   of the columns, and the first row of the input will be read into the first
#'   row of the datset. Note that if the underlying text file also includes
#'   column names in it's first row, this row should be skipped explicitly with
#'   [dataset_skip()].
#'
#'   If `NULL`, the first row of the input will be used as the column names, and
#'   will not be included in dataset.
#'
#' @param col_types Column types. If `NULL` and `col_defaults` is specified
#'   then types will be imputed from the defaults. Otherwise, all column types
#'   will be imputed from the first 1000 rows on the input. This is convenient
#'   (and fast), but not robust. If the imputation fails, you'll need to supply
#'   the correct types yourself.
#'
#'   Types can be explicitliy specified in a character vector as "integer",
#'   "double", and "character" (e.g. `col_types = c("double", "double",
#'   "integer"`).
#'
#'   Alternatively, you can use a compact string representation where each
#'   character represents one column: c = character, i = integer, d = double
#'   (e.g. `col_types = `ddi`).
#'
#' @param col_defaults List of default values which are used when data is
#'   missing from a record (e.g. `list(0, 0, 0L`). If `NULL` then defaults will
#'   be automatically provided based on `col_types` (`0` for numeric columns and
#'   `""` for character columns).
#'
#' @param delim Character delimiter to separate fields in a record (defaults to
#'   ",")
#'
#' @param parallel_records (Optional) An integer, representing the number of
#'   records to decode in parallel. If not specified, records will be
#'   processed sequentially.
#'
#' @importFrom utils read.csv
#'
#' @note The [csv_dataset()] function is a convenience wrapper for the
#'   [text_line_dataset()] and [dataset_decode_delim()] functions.
#'
#' @family dataset methods
#'
#' @export
dataset_decode_delim <- function(dataset, delim = ",",
                                 col_names = NULL, col_types = NULL, col_defaults = NULL,
                                 parallel_records = NULL) {

  # resolve options
  opts <- resolve_dataset_options(dataset, delim, col_names, col_types, col_defaults, 0)

  # read csv
  dataset <- dataset %>%
    dataset_skip(opts$skip) %>%
    dataset_map(
      map_func = function(line) {
        decoded <- tf$decode_csv(
          records = line,
          record_defaults = opts$col_defaults,
          field_delim = delim
        )
        names(decoded) <- opts$col_names
        decoded
      },
      num_parallel_calls = parallel_records
    )

  # return dataset
  as_tf_dataset(dataset)
}


resolve_dataset_options <- function(dataset, delim, col_names, col_types, col_defaults, skip) {

  # if they are all already provied then just reflect them back (along with no skip)
  if (!is.null(col_names) && !is.null(col_types) && !is.null(col_defaults)) {
    return(list(
      col_names = col_names,
      col_types = col_types,
      col_defaults = col_defaults,
      skip = skip
    ))
  }

  # if we are going to need to impute column names/types/defaults then preview
  # the dataset. Note that this will involve evaluating tensors so will make
  # this function not a "pure" graph function (important if you want to include
  # it in dataset_map or dataset_interleave function)
  if (requires_preview(col_names, col_types, col_defaults))
    preview <- preview_dataset(dataset, delim, col_names)
  else
    preview <- NULL

  # validate columns helper (no-op if there is no preview)
  validate_columns <- function(object, name) {
    if (!is.null(preview)) {
      if (length(object) != ncol(preview))
        stop(sprintf(
          "Incorrect number of %s provided (dataset has %d columns)", name, ncol(preview)
        ), call. = FALSE)
    }
  }

  # resolve/validate col_names (add extra skip if we have col_names in the file)
  if (is.null(col_names)) {
    col_names <- names(preview)
    skip <- skip + 1
  } else if (is.character(col_names)) {
    validate_columns(col_names, 'col_names')
  } else {
    stop("col_names must be a character vector")
  }

  # resolve/validate col_types
  if (is.character(col_types)) {

    # convert to lower
    col_types <- tolower(col_types)

    # unpack abbreviations in a single string
    if (length(col_types) == 1 && grepl("^[idc]+$", col_types))
      col_types <- strsplit(col_types, "")[[1]]

    # resolve abbreviations
    col_types <- sapply(tolower(col_types), simplify = TRUE, function(type) {
      switch(type,
             i = "integer",
             d = "double",
             c = "character",
             type
      )
    })

    # validate the type specifiers
    if (!all(col_types %in% c("integer", "double", "character")))
      stop('Invalid column type specification. Valid types are "integer", "double", ',
           'and "character"\n  (or abbreviations "i", "d", and "c").')

    # validate we have the correct number of columns
    validate_columns(col_types, 'col_types')


  } else if (is.null(col_types)) {

    # derive types from col_defaults if provided, otherwise from the preview
    if (!is.null(col_defaults))
      col_types <- sapply(col_defaults, simplify = TRUE, typeof)
    else
      col_types <- sapply(preview, simplify = TRUE, typeof)

    # map into just integer, double, and character
    col_types <- sapply(col_types, simplify = TRUE, function(type) {
      switch(type,
             integer = "integer",
             double = "double",
             character = "character",
             "character" # default
      )
    })
  }

  # resolve/validate record defaults
  if (is.null(col_defaults)) {
    col_defaults <- lapply(unname(col_types), function(type) {
      switch(type,
             integer = list(0L),
             double = list(0),
             character = list(""),
             list("") # default
      )
    })
  } else if (is.list(col_defaults)) {
    validate_columns(col_defaults, 'col_defaults')
    col_defaults <- lapply(col_defaults, function(x) {
      if (!is.list(x))
        list(x)
      else
        x
    })
  } else {
    stop("col_defaults must be NULL (automatic) or a list of default values")
  }

  # return
  list(
    col_names = col_names,
    col_types = col_types,
    col_defaults = col_defaults,
    skip = skip
  )
}


preview_dataset <- function(dataset, delim, col_names) {

  # read the first 1000 rows to faciliate deduction of column names / types as well
  # as checking that any specified col_names, col_types, or col_defaults
  # have the correct length
  preview <- dataset %>%
    dataset_take(1000) %>%
    dataset_batch(1000)
  preview <- with_session(function(session) {
    batch <- next_batch(preview)
    session$run(batch)
  })

  # no-op for empty preview
  if (length(preview) == 0)
    return(dataset)

  # convert bytes to string if necessary
  if (is.list(preview) && inherits(preview[[1]], "python.builtin.bytes")) {
    preview <- unlist(lapply(preview, function(line) line$decode()))
  }

  # read the csv using read.csv to do column/type deduction
  preview_con <- textConnection(preview)
  on.exit(close(preview_con), add = TRUE)
  read.csv(
    file = preview_con,
    header = is.null(col_names) || is.character(col_names),
    sep = delim,
    comment.char = "",
    stringsAsFactors = FALSE
  )
}

read_dataset <- function(filenames, read_func, parallel_files, parallel_files_block_length) {

  # if we are reading files in parallel then use interleave, otherwise flat_map
  if (!is.null(parallel_files)) {
    dataset <- filenames %>%
      dataset_interleave(cycle_length = parallel_files,
                         block_length = parallel_files_block_length,
                         read_func)
  } else {
    dataset <- filenames %>%
      dataset_flat_map(read_func)
  }

  # return dataset
  as_tf_dataset(dataset)
}

requires_preview <- function(col_names, col_types, col_defaults) {
  is.null(col_names) || (is.null(col_types) && is.null(col_defaults))
}






