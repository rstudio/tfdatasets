library(tfdatasets)
library(reticulate)
arr <- function(...) array(seq_len(prod(c(...))), c(...))

X <- arr(50, 3, 3)

x1 <- tfdatasets::tensor_slices_dataset(X)
x2 <- tfdatasets::tensor_slices_dataset(X)
x3 <- tfdatasets::tensor_slices_dataset(X)
ds <- zip_datasets(x1, tuple(x2, x3))

