
library(tfdatasets)


dataset <- csv_dataset("data/mtcars-train.csv", record_defaults = 0) %>%
  dataset_batch(10)

batch <- batch_from_dataset(dataset,
  features = c("disp", "cyl"),
  response = "mpg"
)

sess <- tf$Session()
sess$run(batch)

