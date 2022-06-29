# tfdatasets 2.8.0

- New `dataset_unbatch()`
- New `dataset_group_by_window()`
- New `dataset_take_while()`
- New `as_tensor()` and `as.array()` methods which can be used on TF Datasets with a single element.

# tfdatasets 2.7.0

- Added compatability with Tensorflow version 2.7
- `as_iterator()`, `iter_next()` and `iterate()` are is now reexported from {reticualte}.
- New `as_array_iterator()`, for converting a dataset into an iterable that yields R arrays.
  (`as_iterator()` yields tensorflow tensors)
- New `dataset_bucket_by_sequence_length()`
- New `dataset_rejection_resample()`
- New `dataset_unique()`
- New `choose_from_datasets()`
- `sample_from_datasets()` gains argument `stop_on_empty_dataset`.
- `dataset_batch()` gains arguments `num_parallel_calls` and `deterministic`.
- `dataset_padded_batch()`:
    Fixed error raised when `drop_remainder=TRUE` with recent TF versions.
    Added examples, docs, and tests.
- `dataset_concatenate()` gains `...` and the ability to combine multiple datasets in one call.

# tfdatasets 2.6.0

- New `dataset_options()` for setting and getting dataset options.
- New `length()` method for tensorflow datasets.
- New `dataset_enumerate()`.
- New `random_integer_dataset()`.
- New `dataset_scan()`, a stateful variant of `dataset_map()`.
- New `dataset_snapshot()` for persisting the output of a dataset to disk.
- `range_dataset()` gains a `dtype` argument.
- `dataset_prefetch()` argument `buffer_size` is now optional, defaults to `tf$data$AUTOTUNE`

# tfdatasets 2.4.0

- Fixed problem when saving models with feature specs (#82).

# tfdatasets 1.13.1

* Add `datatset_window` method.
* Allow `purrr` style lambda functions in `dataset_map`.
* Added a `NEWS.md` file to track changes to the package.
* Added a new feature spec interface that can be used to easily create `feature_column`s. (#42)
