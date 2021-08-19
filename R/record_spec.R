



#' Specification for reading a record from a text file with delimited values
#'
#' @param example_file File that provides an example of the records to be read.
#'   If you don't explicitly specify names and types (or defaults) then this
#'   file will be read to generate default values.
#'
#' @param delim Character delimiter to separate fields in a record (defaults to
#'   ",")
#'
#' @param skip Number of lines to skip before reading data. Note that if
#'   `names` is explicitly provided and there are column names witin the
#'   file then `skip` should be set to 1 to ensure that the column names are
#'   bypassed.
#'
#' @param names Character vector with column names (or `NULL` to automatically
#'   detect the column names from the first row of `example_file`).
#'
#'   If `names` is a character vector, the values will be used as the names of
#'   the columns, and the first row of the input will be read into the first row
#'   of the datset. Note that if the underlying text file also includes column
#'   names in it's first row, this row should be skipped explicitly with `skip =
#'   1`.
#'
#'   If `NULL`, the first row of the example_file will be used as the column
#'   names, and will be skipped when reading the dataset.
#'
#' @param types Column types. If `NULL` and `defaults` is specified then types
#'   will be imputed from the defaults. Otherwise, all column types will be
#'   imputed from the first 1000 rows of the `example_file`. This is convenient
#'   (and fast), but not robust. If the imputation fails, you'll need to supply
#'   the correct types yourself.
#'
#'   Types can be explicitliy specified in a character vector as "integer",
#'   "double", and "character" (e.g. `col_types = c("double", "double",
#'   "integer"`).
#'
#'   Alternatively, you can use a compact string representation where each
#'   character represents one column: c = character, i = integer, d = double
#'   (e.g. `types = `ddi`).
#'
#' @param defaults List of default values which are used when data is
#'   missing from a record (e.g. `list(0, 0, 0L`). If `NULL` then defaults will
#'   be automatically provided based on `types` (`0` for numeric columns and
#'   `""` for character columns).
#'
#' @export
delim_record_spec <- function(example_file, delim = ",", skip = 0,
                              names = NULL, types = NULL, defaults = NULL) {

  # if they are all already provided then just reflect spec options back
  if (!is.null(names) && !is.null(types) && !is.null(defaults)) {
    return(tf_dataset_record_spec(
      names = names,
      types = types,
      defaults = defaults,
      delim = delim,
      skip = skip
    ))
  }

  # if we are going to need to impute column names/types/defaults then preview
  if ((is.null(names) || (is.null(types) && is.null(defaults)))) {
    # verify example_file
    if (missing(example_file)) {
      stop("You must provide an example_file if you don't explicitly specify ",
           "names and types (or defaults)")
    }
    # do the preview
    preview <- utils::read.csv(
      file = example_file,
      header = is.null(names) || is.character(names),
      sep = delim,
      nrows = 1000,
      skip = skip,
      blank.lines.skip = FALSE,
      comment.char = "",
      stringsAsFactors = FALSE
    )
  } else {
    preview <- NULL
  }

  # validate columns helper (no-op if there is no preview)
  validate_columns <- function(object, name) {
    if (!is.null(preview)) {
      if (length(object) != ncol(preview))
        stop(sprintf(
          "Incorrect number of %s provided (dataset has %d columns)", name, ncol(preview)
        ), call. = FALSE)
    }
  }

  # resolve/validate names (add extra skip if we have names in the file)
  if (is.null(names)) {
    names <- names(preview)
    skip <- skip + 1
  } else if (is.character(names)) {
    validate_columns(names, 'names')
  } else {
    stop("names must be a character vector")
  }

  # if types is null and defaults is null and we have a preview, then
  # impute from the preview
  if (is.null(types) && is.null(defaults) && !is.null(preview))
    types <- sapply(preview, simplify = TRUE, typeof)

  # final resolution of types and defaults
  types <- resolve_record_types(types, defaults)

  # return
  tf_dataset_record_spec(
    names = names,
    types = types$types,
    defaults = types$defaults,
    delim = delim,
    skip = skip
  )
}

#' @rdname delim_record_spec
#' @export
csv_record_spec <- function(example_file, skip = 0,
                            names = NULL, types = NULL, defaults = NULL) {
  delim_record_spec(example_file, delim = ",", skip, names, types, defaults)
}

#' @rdname delim_record_spec
#' @export
tsv_record_spec <- function(example_file, skip = 0,
                            names = NULL, types = NULL, defaults = NULL) {
  delim_record_spec(example_file, delim = "\t", skip, names, types, defaults)
}

resolve_record_types <- function(types, defaults) {

  # resolve/validate types
  if (is.null(types)) {

    # derive types from defaults
    types <- sapply(defaults, simplify = TRUE, typeof)

    # map into just integer, double, and character
    types <- sapply(types, simplify = TRUE, function(type) {
      switch(type,
             integer = "integer",
             double = "double",
             character = "character",
             "character" # default
      )
    })

  } else {

    # convert to lower
    types <- tolower(types)

    # unpack abbreviations in a single string
    if (length(types) == 1 && grepl("^[idc]+$", types))
      types <- strsplit(types, "")[[1]]

    # resolve abbreviations
    types <- sapply(tolower(types), simplify = TRUE, function(type) {
      switch(type,
             i = "integer",
             d = "double",
             c = "character",
             type
      )
    })

    # validate the type specifiers
    if (!all(types %in% c("integer", "double", "character")))
      stop('Invalid column type specification. Valid types are "integer", "double", ',
           'and "character"\n  (or abbreviations "i", "d", and "c").')
  }

  # resolve/validate record defaults
  if (is.null(defaults)) {

    defaults <- lapply(unname(types), function(type) {
      switch(type,
             integer = list(0L),
             double = list(0),
             character = list(""),
             list("") # default
      )
    })

  } else {

    defaults <- lapply(defaults, function(x) {
      if (!is.list(x))
        list(x)
      else
        x
    })

  }

  list(
    types = types,
    defaults = defaults
  )

}


#' @rdname sql_dataset
#' @export
sql_record_spec <- function(names, types) {
  structure(class = "tf_dataset_record_spec", list(
    names = names,
    types = types,
    defaults = NULL,
    delim = NULL,
    skip = NULL
  ))
}


tf_dataset_record_spec <- function(names, types, defaults, delim, skip) {
  structure(class = "tf_dataset_record_spec", list(
    names = names,
    types = types,
    defaults = defaults,
    delim = delim,
    skip = skip
  ))
}
