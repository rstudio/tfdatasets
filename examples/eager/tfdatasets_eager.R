
library(tensorflow)
tfe_enable_eager_execution()

library(tfdatasets)

record_spec <- sql_record_spec(
  names = c("disp", "drat", "vs", "gear", "mpg", "qsec", "hp", "am", "wt","carb", "cyl"),
  types = c(tf$float64, tf$int32, tf$float64, tf$int32, tf$float64, tf$float64,
            tf$float64, tf$int32, tf$int32, tf$int32, tf$int32)
)

dataset <- sqlite_dataset('mtcars.sqlite3', 'select * from mtcars', record_spec) %>%
  dataset_batch(10)

dataset %>% for_each_batch(function(batch) {
  str(batch)
})

