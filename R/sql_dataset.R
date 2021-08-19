
#' A dataset consisting of the results from a SQL query
#'
#' @param driver_name String containing the database type. Currently, the only
#'   supported value is 'sqlite'.
#' @param data_source_name String containing a connection string to connect to
#'   the database.
#' @param filename Filename for the database
#' @param query String containing the SQL query to execute.
#' @param record_spec Names and types of database columns
#' @param names Names of columns returned from the query
#' @param types List of `tf$DType` objects (e.g. `tf$int32`,
#'   `tf$double`, `tf$string`) representing the types of the columns
#'   returned by the query.
#'
#' @return A dataset
#'
#' @export
sql_dataset <- function(driver_name, data_source_name, query, record_spec) {

  # validate during dataset construction
  validate_tf_version("1.7", "sql_dataset")

  # dataset
  dataset <- tfd_SqlDataset(
    driver_name,
    data_source_name,
    query,
    tuple(record_spec$types)
  )

  # convert to record
  dataset <- dataset_map(dataset, function(...) {
    record <- list(...)
    if (!is.null(record_spec$names))
      names(record) <- record_spec$names
    record
  })

  # return dataset
  as_tf_dataset(dataset)
}



#' @rdname sql_dataset
#' @export
sqlite_dataset <- function(filename, query, record_spec) {
  if (is.character(filename))
    filename <- path.expand(filename)
  sql_dataset("sqlite", filename, query, record_spec)
}
