


#' Tensor(s) for retrieving the next batch from a dataset
#'
#' @param dataset A dataset
#'
#' @return Tensor(s) that can be evaluated to yield the next batch of training data.
#'
#' @details
#'
#' To access the underlying data within the dataset you iteratively evaluate the
#' tensor(s) to read batches of data.
#'
#' Note that in many cases you won't need to explicitly evaluate the tensors.
#' Rather, you will pass the tensors to another function that will perform
#' the evaluation (e.g. the Keras [layer_input()][keras::layer_input()] and
#' [compile()][keras::compile()] functions).
#'
#' If you do need to perform iteration manually by evaluating the tensors, there
#' are a couple of possible approaches to controlling/detecting when iteration should
#' end.
#'
#' One approach is to create a dataset that yields batches infinitely (traversing
#' the dataset multiple times with different batches randomly drawn). In this case you'd
#' use another mechanism like a global step counter or detecting a learning plateau.
#'
#' Another approach is to detect when all batches have been yielded
#' from the dataset. When the tensor reaches the end of iteration a runtime
#' error will occur. You can catch and ignore the error when it occurs by wrapping
#' your iteration code in the `with_dataset()` function.
#'
#' See the examples below for a demonstration of each of these methods of iteration.
#'
#' @examples \dontrun{
#'
#' # iteration with 'infinite' dataset and explicit step counter
#'
#' library(tfdatasets)
#' dataset <- text_line_dataset("mtcars.csv", record_spec = mtcars_spec) %>%
#'   dataset_prepare(x = c(mpg, disp), y = cyl) %>%
#'   dataset_shuffle(5000) %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat() # repeat infinitely
#' batch <- next_batch(dataset)
#' steps <- 200
#' for (i in 1:steps) {
#'   # use batch$x and batch$y tensors
#' }
#'
#' # iteration that detects and ignores end of iteration error
#'
#' library(tfdatasets)
#' dataset <- text_line_dataset("mtcars.csv", record_spec = mtcars_spec) %>%
#'   dataset_prepare(x = c(mpg, disp), y = cyl) %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat(10)
#' batch <- next_batch(dataset)
#' with_dataset({
#'   while(TRUE) {
#'     # use batch$x and batch$y tensors
#'   }
#' })
#' }
#'
#' @export
next_batch <- function(dataset) {

  # get the iterator
  iter <- make_iterator_one_shot(dataset)
  next_batch <- iter$get_next()

  # re-arrange x and y if necessary
  if (identical(names(next_batch), c("y", "x")))
    next_batch <- list(x = next_batch[["x"]], y = next_batch[["y"]])

  # return
  next_batch
}



#' Execute code that traverses a dataset
#'
#' @param expr Expression to execute
#'
#' @details  When a dataset iterator reaches the end, an out of range runtime error
#'   will occur. You can catch and ignore the error when it occurs by wrapping
#'   your iteration code in a call to `with_dataset()` (see the example
#'   below for an illustration).
#'
#' @examples \dontrun{
#' library(tfdatasets)
#' dataset <- text_line_dataset("mtcars.csv", record_spec = mtcars_spec) %>%
#'   dataset_prepare(x = c(mpg, disp), y = cyl) %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat(10)
#'
#' iter <- make_iterator_one_shot(dataset)
#' next_batch <- iterator_get_next(iter)
#'
#' with_dataset({
#'   while(TRUE) {
#'     batch <- sess$run(next_batch)
#'     # use batch$x and batch$y tensors
#'   }
#' })
#' }
#'
#' @export
with_dataset <- function(expr) {
  tryCatch({
    force(expr)
  },
  error = out_of_range_handler)
}

#' Execute code that traverses a dataset until an out of range condition occurs
#'
#' @param expr Expression to execute (will be executed multiple times until
#' the condition occurs)
#' @param e Error object
#'
#' @details  When a dataset iterator reaches the end, an out of range runtime error
#'   will occur. This function will catch and ignore the error when it occurs.
#'
#' @examples \dontrun{
#' library(tfdatasets)
#' dataset <- text_line_dataset("mtcars.csv", record_spec = mtcars_spec) %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat(10) %>%
#'   dataset_prepare(x = c(mpg, disp), y = cyl)
#'
#' iter <- make_iterator_one_shot(dataset)
#' next_batch <- iterator_get_next(iter)
#'
#' until_out_of_range({
#'   batch <- sess$run(next_batch)
#'   # use batch$x and batch$y tensors
#' })
#' }
#'
#' @family reading datasets
#'
#' @export
until_out_of_range <- function(expr) {

  # determine the error message for 'break'
  break_error <- tryCatch(eval(parse(text = "break")), error = function(e) e)

  # get expression and parent frame
  expr <- substitute(expr)
  envir <- parent.frame()

  # evaluate repeatedly (exit gracefully on break or out_of_range error)
  tryCatch({
    while(TRUE)
      eval(expr, envir = envir)
  },
  error = function(e) {
    if (!identical(e$message, break_error$message))
      out_of_range_handler(e)
  })
}


#' @rdname until_out_of_range
#' @export
out_of_range_handler <- function(e) {
  last_error <- py_last_error()
  if (is.null(last_error) || !identical(last_error$type, "OutOfRangeError"))
    stop(e$message, call. = FALSE)
}
