# DIINA project-grounded analysis. Run from this file's directory or set DIINA_PROJECT_DIR.
find_project <- function() {
  candidates <- c(Sys.getenv("DIINA_PROJECT_DIR", ""), ".", "..", "../project_inputs", "project_inputs", "/mnt/data")
  candidates <- candidates[nzchar(candidates)]
  for (p in candidates) if (file.exists(file.path(p, "raw_results.csv"))) return(normalizePath(p))
  stop("Could not find raw_results.csv; set DIINA_PROJECT_DIR to project_inputs or source directory.")
}
project_dir <- find_project()
read_telemetry <- function() read.csv(file.path(project_dir, "raw_results.csv"), stringsAsFactors=FALSE)
# Fast/slow path utilization from step 1 to 300 by seed; values are proportions.
d <- read_telemetry(); by_seed <- do.call(rbind,lapply(split(d,d$seed),function(x){x<-x[order(x$step),];data.frame(seed=x$seed[1],fast_initial=x$fast_usage[1],fast_final=x$fast_usage[nrow(x)],fast_shift=x$fast_usage[nrow(x)]-x$fast_usage[1],slow_initial=x$slow_usage[1],slow_final=x$slow_usage[nrow(x)],slow_shift=x$slow_usage[nrow(x)]-x$slow_usage[1])}))
print(by_seed); cat("Mean fast shift:",mean(by_seed$fast_shift),"; mean slow shift:",mean(by_seed$slow_shift),"\n")
cat("All-step means:",mean(d$fast_usage),mean(d$slow_usage),"\n")
