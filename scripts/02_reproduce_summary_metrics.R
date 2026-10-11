# DIINA project-grounded analysis. Run from this file's directory or set DIINA_PROJECT_DIR.
find_project <- function() {
  candidates <- c(Sys.getenv("DIINA_PROJECT_DIR", ""), ".", "..", "../project_inputs", "project_inputs", "/mnt/data")
  candidates <- candidates[nzchar(candidates)]
  for (p in candidates) if (file.exists(file.path(p, "raw_results.csv"))) return(normalizePath(p))
  stop("Could not find raw_results.csv; set DIINA_PROJECT_DIR to project_inputs or source directory.")
}
project_dir <- find_project()
read_telemetry <- function() read.csv(file.path(project_dir, "raw_results.csv"), stringsAsFactors=FALSE)
# Recompute seed-level means and final values; compare exact JSON summary fields.
if (!requireNamespace("jsonlite",quietly=TRUE)) stop("Install jsonlite")
d <- read_telemetry(); s <- jsonlite::fromJSON(file.path(project_dir,"summary.json"))
by_seed <- do.call(rbind,lapply(split(d,d$seed),function(x) data.frame(seed=x$seed[1], avg_loss=mean(x$loss), final_loss=x$loss[which.max(x$step)], stringsAsFactors=FALSE)))
stat <- function(x) c(mean=mean(x), sd=sd(x), ci95=qt(.975,length(x)-1)*sd(x)/sqrt(length(x)))
print(by_seed); cat("Seed-level metric summary (sample SD; t CI):\n"); print(rbind(avg_loss=stat(by_seed$avg_loss),final_loss=stat(by_seed$final_loss)))
cat("JSON avg_loss:",unlist(s$avg_loss),"\nJSON final_loss:",unlist(s$final_loss),"\n")
cat("Differences [empirical minus JSON]:\n")
print(c(avg_mean=mean(by_seed$avg_loss)-s$avg_loss$mean,avg_sd=sd(by_seed$avg_loss)-s$avg_loss$std,avg_ci=stat(by_seed$avg_loss)[3]-s$avg_loss$ci95,final_mean=mean(by_seed$final_loss)-s$final_loss$mean,final_sd=sd(by_seed$final_loss)-s$final_loss$std,final_ci=stat(by_seed$final_loss)[3]-s$final_loss$ci95))
traj <- aggregate(loss~step,d, function(x)c(mean=mean(x),sd=sd(x),n=length(x)))
cat("Trajectory first/last rows (mean, SD, n):\n"); print(head(traj,3)); print(tail(traj,3))
