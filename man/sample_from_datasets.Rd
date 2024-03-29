% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sample_from_datasets.R
\name{sample_from_datasets}
\alias{sample_from_datasets}
\title{Samples elements at random from the datasets in \code{datasets}.}
\usage{
sample_from_datasets(
  datasets,
  weights = NULL,
  seed = NULL,
  stop_on_empty_dataset = TRUE
)
}
\arguments{
\item{datasets}{A list ofobjects with compatible structure.}

\item{weights}{(Optional.) A list of \code{length(datasets)} floating-point values where
\code{weights[[i]]} represents the probability with which an element should be sampled
from \code{datasets[[i]]}, or a dataset object where each element is such a list.
Defaults to a uniform distribution across \code{datasets}.}

\item{seed}{(Optional.) An integer, representing the random seed
that will be used to create the distribution.}

\item{stop_on_empty_dataset}{If \code{TRUE}, selection stops if it encounters an
empty dataset. If \code{FALSE}, it skips empty datasets. It is recommended to
set it to \code{TRUE}. Otherwise, the selected elements start off as the user
intends, but may change as input datasets become empty. This can be
difficult to detect since the dataset starts off looking correct. Defaults
to \code{TRUE}.}
}
\value{
A dataset that interleaves elements from \code{datasets} at random, according to
\code{weights} if provided, otherwise with uniform probability.
}
\description{
Samples elements at random from the datasets in \code{datasets}.
}
