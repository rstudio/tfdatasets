library(tfdatasets)
library(tfestimators)

mtcars_spec <- csv_record_spec("mtcars-train.csv")


# dataset w/ batch size of 10 that repeats for 5 epochs
dataset <- text_line_dataset("mtcars-train.csv", record_spec = mtcars_spec) %>%
  dataset_shuffle(100) %>%
  dataset_batch(30) %>%
  dataset_repeat(5)

