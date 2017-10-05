


as_integer_tensor <- function(x, dtype = tf$int64) {
  if (is.null(x))
    x
  else if (inherits(x, "tensorflow.python.framework.ops.Tensor"))
    tf$cast(x, dtype = dtype)
  else
    tf$constant(as.integer(x), dtype = dtype)
}


