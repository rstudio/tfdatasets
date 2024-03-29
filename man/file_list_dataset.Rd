% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/file_list_dataset.R
\name{file_list_dataset}
\alias{file_list_dataset}
\title{A dataset of all files matching a pattern}
\usage{
file_list_dataset(file_pattern, shuffle = NULL, seed = NULL)
}
\arguments{
\item{file_pattern}{A string, representing the filename pattern that will be matched.}

\item{shuffle}{(Optional) If \code{TRUE}, the file names will be shuffled randomly.
Defaults to \code{TRUE}}

\item{seed}{(Optional) An integer, representing the random seed that
will be used to create the distribution.}
}
\value{
A dataset of string corresponding to file names
}
\description{
A dataset of all files matching a pattern
}
\details{
For example, if we had the following files on our filesystem:
\itemize{
\item /path/to/dir/a.txt
\item /path/to/dir/b.csv
\item /path/to/dir/c.csv
}

If we pass \code{"/path/to/dir/*.csv"} as the \code{file_pattern}, the dataset would produce:
\itemize{
\item /path/to/dir/b.csv
\item /path/to/dir/c.csv
}
}
\note{
The \code{shuffle} and \code{seed} arguments only apply for TensorFlow >= v1.8
}
