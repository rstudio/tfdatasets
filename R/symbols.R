# Locations of symbols in TF 1.x vs TF 2.x

# tf --> tf$random
tfr_random_uniform <- function(...) if (tensorflow::tf_version() < "1.14") tf$random_uniform(...) else tf$random$uniform(...)

# tf --> tf$io
tfio_decode_csv <- function(...) if (tensorflow::tf_version() < "1.14") tf$decode_csv(...) else tf$io$decode_csv(...)

# tf$contrib$data --> tf$data$experimental
tfd_make_csv_dataset <- function(...) if (tensorflow::tf_version() < "1.14") tf$contrib$data$make_csv_dataset(...) else tf$data$experimental$make_csv_dataset(...)
tfd_shuffle_and_repeat <- function(...) if (tensorflow::tf_version() < "1.14") tf$contrib$data$shuffle_and_repeat(...) else tf$data$experimental$shuffle_and_repeat(...)
tfd_map_and_batch <- function(...) if (tensorflow::tf_version() < "1.14") tf$contrib$data$map_and_batch(...) else tf$data$experimental$map_and_batch(...)
tfd_prefetch_to_device <- function(...) if (tensorflow::tf_version() < "1.14") tf$contrib$data$prefetch_to_device(...) else tf$data$experimental$prefetch_to_device(...)
tfd_SqlDataset <- function(...) if (tensorflow::tf_version() < "1.14") tf$contrib$data$SqlDataset(...) else tf$data$experimental$SqlDataset(...)
