% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dataset_methods.R
\name{dataset_scan}
\alias{dataset_scan}
\title{A transformation that scans a function across an input dataset}
\usage{
dataset_scan(dataset, initial_state, scan_func)
}
\arguments{
\item{dataset}{A tensorflow dataset}

\item{initial_state}{A nested structure of tensors, representing the initial
state of the accumulator.}

\item{scan_func}{A function that maps \verb{(old_state, input_element)} to
\verb{(new_state, output_element)}. It must take two arguments and return a
pair of nested structures of tensors. The \code{new_state} must match the
structure of \code{initial_state}.}
}
\description{
A transformation that scans a function across an input dataset
}
\details{
This transformation is a stateful relative of \code{dataset_map()}.
In addition to mapping \code{scan_func} across the elements of the input dataset,
\code{scan()} accumulates one or more state tensors, whose initial values are
\code{initial_state}.
}
\examples{
\dontrun{
initial_state <- as_tensor(0, dtype="int64")
scan_func <- function(state, i) list(state + i, state + i)
dataset <- range_dataset(0, 10) \%>\%
  dataset_scan(initial_state, scan_func)

reticulate::iterate(dataset, as.array) \%>\%
  unlist()
# 0  1  3  6 10 15 21 28 36 45
}
}
