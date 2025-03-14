% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dataset_methods.R
\name{dataset_collect}
\alias{dataset_collect}
\title{Collects a dataset}
\usage{
dataset_collect(dataset, iter_max = Inf)
}
\arguments{
\item{dataset}{A dataset}

\item{iter_max}{Maximum number of iterations. \code{Inf} until the end of the
dataset}
}
\description{
Iterates throught the dataset collecting every element into a list.
It's useful for looking at the full result of the dataset.
Note: You may run out of memory if your dataset is too big.
}
\seealso{
Other dataset methods: 
\code{\link{dataset_batch}()},
\code{\link{dataset_cache}()},
\code{\link{dataset_concatenate}()},
\code{\link{dataset_decode_delim}()},
\code{\link{dataset_filter}()},
\code{\link{dataset_interleave}()},
\code{\link{dataset_map}()},
\code{\link{dataset_map_and_batch}()},
\code{\link{dataset_padded_batch}()},
\code{\link{dataset_prefetch}()},
\code{\link{dataset_prefetch_to_device}()},
\code{\link{dataset_rebatch}()},
\code{\link{dataset_reduce}()},
\code{\link{dataset_repeat}()},
\code{\link{dataset_shuffle}()},
\code{\link{dataset_shuffle_and_repeat}()},
\code{\link{dataset_skip}()},
\code{\link{dataset_take}()},
\code{\link{dataset_take_while}()},
\code{\link{dataset_window}()}
}
\concept{dataset methods}
