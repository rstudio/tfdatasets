
## R interface to TensorFlow Dataset API

[![Travis-CI Build Status](https://travis-ci.org/rstudio/tfdatasets.svg?branch=master)](https://travis-ci.org/rstudio/tfdatasets) 
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/rstudio/tfdatasets?branch=master&svg=true)](https://ci.appveyor.com/project/rstudio/tfdatasets)

The TensorFlow Dataset API provides various facilities for creating scalable input pipelines for TensorFlow models, including:

- Reading data from a variety of formats including CSV files and [TFRecords files](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) (the standard binary format for TensorFlow training data).

- Transforming datasets in a variety of ways including mapping arbitrary functions against them.

- Shuffling, batching, and repeating datasets over a number of epochs.

- Streaming interface to data for reading arbitrarily large datasets.

- Reading and transforming data are TensorFlow graph operations, so are executed in C++ and in parallel with model training.

The R interface to TensorFlow datasets provides access to the Dataset API, including high-level convenience functions for easy integration with the [tfestimators](https://tensorflow.rstudio.com/tfestimators) package.

For documentation on using tfdatasets, see the package website at <https://tensorflow.rstudio.com/tools/tfdatasets/>.
