- dataset_decode_delim
   - csv_dataset
   - delim_dataset


- col_types as character vector ("integer", "numeric", "character") or "INC" as readr (also look at readxl type specifier)

- field_delim becomes delim

- dataset_recipe


dataset_prepare <- function(dataset, x, y = NULL,
                            named = TRUE, named_features = FALSE) {

}


batch_tensor
input_fn.dataset


batch_from_dataset(dataset) (make an iterator and get it's tensor)

separate iterator functions as necessary
  - one_shot_iterator()
  - iterator_get_next()



