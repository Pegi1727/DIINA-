# DIINA project-grounded analysis. Run from this file's directory or set DIINA_PROJECT_DIR.
find_project <- function() {
  candidates <- c(Sys.getenv("DIINA_PROJECT_DIR", ""), ".", "..", "../project_inputs", "project_inputs", "/mnt/data")
  candidates <- candidates[nzchar(candidates)]
  for (p in candidates) if (file.exists(file.path(p, "raw_results.csv"))) return(normalizePath(p))
  stop("Could not find raw_results.csv; set DIINA_PROJECT_DIR to project_inputs or source directory.")
}
project_dir <- find_project()
read_telemetry <- function() read.csv(file.path(project_dir, "raw_results.csv"), stringsAsFactors=FALSE)
# Factor grid as coded in v18: lengths {10,50,100,200}, noise {0,.1,.2}.
# Educational statistical degradation model (not reported model accuracy): transparent proxy curve.
lengths <- c(10,50,100,200); noises <- c(0,.1,.2); grid <- expand.grid(length=lengths,noise=noises)
# Error proxy combines token corruption and a length penalty; its formula is illustrative, not logged evidence.
grid$illustrative_error <- pmin(1, grid$noise + 0.015*log1p(grid$length/10))
grid$illustrative_accuracy <- 1-grid$illustrative_error
print(grid); cat("v18 has",nrow(grid),"factor conditions; source trains/evaluates Transformer and gated-GRU DIINA per cell.\n")
