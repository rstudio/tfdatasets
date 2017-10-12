
library(tfdatasets)


dataset <- csv_dataset("data/mtcars-train.csv", record_defaults = 0) %>%
  dataset_batch(10)

batches <- batches_from_dataset(dataset,
  features = c("disp", "cyl"),
  response = "mpg",
  named = FALSE
)

next_element_tensor(iterator)
