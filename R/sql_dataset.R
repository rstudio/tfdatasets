
#' A dataset consisting of the results from a SQL query
#'
#' @param driver_name String containing the database type. Currently, the only
#'   supported value is 'sqlite'.
#' @param data_source_name String containing a connection string to connect to
#'   the database.
#' @param filename Filename for the database
#' @param query String containing the SQL query to execute.
#' @param output_types List of `tf$DType`` objects (e.g. `tf$int32`,
#'   `tf$float32`) representing the types of the columns returned by query.
#'
#' @return A dataset
#'
#' @export
sql_dataset <- function(driver_name, data_source_name, query, output_types) {

  # validate during dataset construction
  validate_tf_version("1.7", "sql_dataset")

  # read the dataset
  as_tf_dataset(
    tf$contrib$data$SqlDataset(
      driver_name,
      data_source_name,
      query,
      tuple(output_types)
    )
  )
}



#' @rdname sql_dataset
#' @export
sqlite_dataset <- function(filename, query, output_types) {
  if (is.character(filename))
    filename <- path.expand(filename)
  sql_dataset("sqlite", filename, query, output_types)
}

