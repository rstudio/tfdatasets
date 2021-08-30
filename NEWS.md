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
