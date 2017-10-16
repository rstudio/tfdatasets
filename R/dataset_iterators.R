


#' Tensor(s) for retreiving the next element from a dataset
#'
#' @param x A dataset or an iterator
#'
#' @family iterators
#'
#' @return Tensor(s) that can be evaluated to yield the next batch of training data.
#'
#' @section Batch Iteration:
#'
#' In many cases you won't need to explicitly evaluate the tensors,
#' rather, you will pass the tensors to another function that will perform
#' the evaluation (e.g. the Keras `layer_input()` and `compile()` functions).
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
#' your iteration code in the `with_dataset_iterator()` function.
#'
#' See the examples below for a demonstration of each of these methods of iteration.
#'
#' @examples \dontrun{
#'
#' # iteration with 'infinite' dataset and explicit step counter
#'
#' library(tfdatasets)
#' dataset <- csv_dataset("training.csv") %>%
#'   dataset_prepare(x = c(mpg, disp), y = cyl) %>%
#'   dataset_shuffle(5000) %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat()
#' batch <- iterator_get_next(dataset)
#' steps <- 200
#' for (i in 1:steps) {
#'   # use batch$x and batch$y tensors
#' }
#'
#' # iteration that detects and ignores end of iteration error
#'
#' library(tfdatasets)
#' dataset <- csv_dataset("training.csv") %>%
#'   dataset_prepare(x = c(mpg, disp), y = cyl) %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat(10)
#' batch <- iterator_get_next(dataset)
#' with_dataset_iterator({
#'   while(TRUE) {
#'     # use batch$x and batch$y tensors
#'   }
#' })
#' }
#'
#' @family iterators
#'
#' @export
iterator_get_next <- function(x) {

  # if it's a dataset then create a one-shot iterator from it
  if (inherits(x, "tensorflow.python.data.ops.dataset_ops.Dataset"))
    x <- x$make_one_shot_iterator()

  # return get_next tensor(s)
  x$get_next()

}



#' Execute code that checks for end of dataset iteration
#'
#' @param expr Expression to execute
#'
#' @details  When a dataset iterator reaches the end, an out of range runtime error
#'   will occur. You can catch and ignore the error when it occurs by wrapping
#'   your iteration code in the `with_dataset_iterator()` (see the example
#'   below for an illustration).
#'
#' @examples \dontrun{
#' library(tfdatasets)
#' dataset <- csv_dataset("training.csv") %>%
#'   dataset_batch(128) %>%
#'   dataset_repeat(10)
#'
#' batch <- batch_from_dataset(dataset, features = c(mpg, disp), response = cyl)
#' with_dataset_iterator({
#'   while(TRUE) {
#'     # use batch$x and batch$y tensors
#'   }
#' })
#' }
#'
#' @family iterators
#'
#' @export
with_dataset_iterator <- function(expr) {
  tryCatch({
    force(expr)
  },
  error = function(e) {
    last_error <- py_last_error()
    if (is.null(last_error) || !identical(last_error$type, "OutOfRangeError"))
      stop(e)
  })
}









