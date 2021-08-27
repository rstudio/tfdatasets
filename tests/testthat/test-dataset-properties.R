
context("dataset properties")

test_succeeds("output_types returns types", {

  dataset <- tensors_dataset(tf$constant(1:100))
  expect_equal(output_types(dataset), tf$int32)

  dataset <- csv_dataset(testing_data_filepath("mtcars.csv"))
  types <- output_types(dataset)
  expect_equal(types$cyl, tf$int32)
  expect_equal(types$mpg, tf$float32)
})

test_succeeds("output_shapes returns shapes", {
  dataset <- tensors_dataset(tf$constant(1:100))
  expect_equal(output_shapes(dataset)$as_list(), 100)

  dataset <- csv_dataset(testing_data_filepath("mtcars.csv"))
  shapes <- output_shapes(dataset)
  expect_equal(shapes$cyl$as_list(), list())
  expect_equal(shapes$mpg$as_list(), list())
})
