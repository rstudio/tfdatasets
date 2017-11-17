library(tfdatasets)
library(tfestimators)

mtcars_spec <- csv_record_spec("mtcars-train.csv")

# return an input_fn for a set of csv files
mtcars_input_fn <- function(filenames) {

  # dataset w/ batch size of 10 that repeats for 5 epochs
  dataset <- text_line_dataset(filenames, record_spec = mtcars_spec) %>%
    dataset_shuffle(20) %>%
    dataset_batch(10) %>%
    dataset_repeat(5)

  # create input_fn from dataset
  input_fn(dataset, features = c("disp", "cyl"), response = "mpg")
}

# define feature columns
cols <- feature_columns(
  column_numeric("disp"),
  column_numeric("cyl")
)

# create model
model <- linear_regressor(feature_columns = cols)

# train model
model %>% train(mtcars_input_fn("mtcars-train.csv"))

# evaluate model
model %>% evaluate(mtcars_input_fn("mtcars-test.csv"))







