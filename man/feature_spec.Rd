% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/feature_spec.R
\name{feature_spec}
\alias{feature_spec}
\title{Creates a feature specification.}
\usage{
feature_spec(dataset, x, y = NULL)
}
\arguments{
\item{dataset}{A TensorFlow dataset.}

\item{x}{Features to include can use \code{\link[tidyselect:language]{tidyselect::select_helpers()}} or
a \code{formula}.}

\item{y}{(Optional) The response variable. Can also be specified using
a \code{formula} in the \code{x} argument.}
}
\value{
a \code{FeatureSpec} object.
}
\description{
Used to create initialize a feature columns specification.
}
\details{
After creating the \code{feature_spec} object you can add steps using the
\code{step} functions.
}
\examples{
\dontrun{
library(tfdatasets)
data(hearts)
hearts <- tensor_slices_dataset(hearts) \%>\% dataset_batch(32)

# use the formula interface
spec <- feature_spec(hearts, target ~ .)

# select using `tidyselect` helpers
spec <- feature_spec(hearts, x = c(thal, age), y = target)
}
}
\seealso{
\itemize{
\item \code{\link[=fit.FeatureSpec]{fit.FeatureSpec()}} to fit the FeatureSpec
\item \code{\link[=dataset_use_spec]{dataset_use_spec()}} to create a tensorflow dataset prepared to modeling.
\item \link{steps} to a list of all implemented steps.
}

Other Feature Spec Functions: 
\code{\link{dataset_use_spec}()},
\code{\link{fit.FeatureSpec}()},
\code{\link{step_bucketized_column}()},
\code{\link{step_categorical_column_with_hash_bucket}()},
\code{\link{step_categorical_column_with_identity}()},
\code{\link{step_categorical_column_with_vocabulary_file}()},
\code{\link{step_categorical_column_with_vocabulary_list}()},
\code{\link{step_crossed_column}()},
\code{\link{step_embedding_column}()},
\code{\link{step_indicator_column}()},
\code{\link{step_numeric_column}()},
\code{\link{step_remove_column}()},
\code{\link{step_shared_embeddings_column}()},
\code{\link{steps}}
}
\concept{Feature Spec Functions}
