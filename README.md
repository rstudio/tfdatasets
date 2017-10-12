
## tfdatasets: R interface to TensorFlow Datasets API

The TensorFlow Datasets API provides various facilities for creating scalable input pipelines for TensorFlow models, including:

- Reading data from a variety of formats including CSV files and [TFRecords files](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) (the standard binary format for TensorFlow training data).

- Transforming datasets in a variety of ways including mapping arbitrary functions against them.

- Shuffling, batching, and repeating datasets over a number of epochs.

- Streaming interface to data for reading arbitrarily large datasets.

- Reading and transforming data are TensorFlow graph operations, so are executed in C++ and in parallel with model training.

The R interface to TensorFlow datasets provides access to the Dataset API, including high-level convenience functions for easy integration with the [keras](https://tensorflow.rstudio.com/keras) and [tfestimators](https://tensorflow.rstudio.com/tfestimators) R packages.

For documentation on using tfestimators, see the package website at https://rstudio.github.io/tfdatasets.
