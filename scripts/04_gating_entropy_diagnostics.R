# DIINA project-grounded analysis. Run from this file's directory or set DIINA_PROJECT_DIR.
find_project <- function() {
  candidates <- c(Sys.getenv("DIINA_PROJECT_DIR", ""), ".", "..", "../project_inputs", "project_inputs", "/mnt/data")
  candidates <- candidates[nzchar(candidates)]
  for (p in candidates) if (file.exists(file.path(p, "raw_results.csv"))) return(normalizePath(p))
  stop("Could not find raw_results.csv; set DIINA_PROJECT_DIR to project_inputs or source directory.")
}
project_dir <- find_project()
read_telemetry <- function() read.csv(file.path(project_dir, "raw_results.csv"), stringsAsFactors=FALSE)
# Entropy floor/ceiling and association with loss; not a causal test.
d <- read_telemetry(); eps <- 1e-12
# Binary normalized entropy upper bound is log(2); logs are retained as observed telemetry.
cat("Entropy range:",range(d$entropy)," mean:",mean(d$entropy)," SD:",sd(d$entropy),"\n")
cat("Distance to binary-collapse value 0:",min(d$entropy),"; distance to log(2):",log(2)-max(d$entropy),"\n")
cat("Pooled correlation entropy-loss:",cor(d$entropy,d$loss),"\n")
per_seed <- do.call(rbind,lapply(split(d,d$seed),function(x)data.frame(seed=x$seed[1],cor_entropy_loss=cor(x$entropy,x$loss),entropy_sd=sd(x$entropy),entropy_min=min(x$entropy))))
print(per_seed); cat("Entropy all finite and strictly above zero:",all(is.finite(d$entropy)&d$entropy>eps),"\n")
