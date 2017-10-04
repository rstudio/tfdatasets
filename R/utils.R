


as_tensor_int64 <- function(x) {
  if (is.null(x))
    x
  else if (inherits(x, "tensorflow.python.framework.ops.Tensor"))
    tf$cast(x, dtype = tf$int64)
  else
    tf$constant(as.integer(x), dtype = tf$int64)
}
