

#' @importFrom reticulate py_last_error tuple py_str py_has_attr
#' @import tidyselect
#' @import rlang
NULL


.onLoad <- function(libname, pkgname) {
  registerMethods(list(
    # c(package, genname, class)
    c("tfestimators", "input_fn", "tf_dataset")
  ))
}


# Reusable function for registering a set of methods with S3 manually. The
# methods argument is a list of character vectors, each of which has the form
# c(package, genname, class).
registerMethods <- function(methods) {
  lapply(methods, function(method) {
    pkg <- method[[1]]
    generic <- method[[2]]
    class <- method[[3]]
    func <- get(paste(generic, class, sep="."))
    if (pkg %in% loadedNamespaces()) {
      registerS3method(generic, class, func, envir = asNamespace(pkg))
    }
    setHook(
      packageEvent(pkg, "onLoad"),
      function(...) {
        registerS3method(generic, class, func, envir = asNamespace(pkg))
      }
    )
  })
}

