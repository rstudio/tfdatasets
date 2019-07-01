#' Heart Disease Data Set
#'
#' Heart disease (angiographic disease status) dataset.
#'
#' @references
#' The authors of the databases have requested that any publications resulting
#' from the use of the data include the names of the principal investigator
#' responsible for the data collection at each institution. They would be:
#'
#' 1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
#' 2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
#' 3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
#' 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:Robert Detrano, M.D., Ph.D.
#'
#' @format A data frame with 303 rows and 14 variables:
#' \describe{
#'   \item{age}{age in years}
#'   \item{sex}{sex (1 = male; 0 = female)}
#'   \item{cp}{chest pain type: Value 1: typical angina, Value 2: atypical angina,
#'   Value 3: non-anginal pain, Value 4: asymptomatic}
#'   \item{trestbps}{resting blood pressure (in mm Hg on admission to the hospital)}
#'   \item{chol}{ serum cholestoral in mg/dl }
#'   \item{fbs}{(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)}
#'   \item{restecg}{resting electrocardiographic results: Value 0: normal, Value 1:
#'   having ST-T wave abnormality (T wave inversions and/or ST elevation or
#'   depression of > 0.05 mV), Value 2: showing probable or definite left ventricular
#'   hypertrophy by Estes' criteria}
#'   \item{thalach}{maximum heart rate achieved}
#'   \item{exang}{exercise induced angina (1 = yes; 0 = no)}
#'   \item{oldpeak}{ST depression induced by exercise relative to rest}
#'   \item{slope}{the slope of the peak exercise ST segment: Value 1: upsloping,
#'   Value 2: flat, Value 3: downsloping}
#'   \item{ca}{number of major vessels (0-3) colored by flourosopy}
#'   \item{thal}{3 = normal; 6 = fixed defect; 7 = reversable defect}
#'   \item{target}{diagnosis of heart disease angiographic}
#' }
#'
#' @source \url{https://archive.ics.uci.edu/ml/datasets/heart+Disease}
"hearts"
