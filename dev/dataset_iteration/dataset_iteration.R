
library(tfdatasets)


dataset <- csv_dataset("data/mtcars-train.csv", record_defaults = "numeric") %>%
  dataset_batch(10)

iterator <- iterator_from_dataset(dataset,
  features = c("disp", "cyl"),
  response = "mpg",
  named_features = FALSE
)

next_element_tensor(iterator)
