# DIINA project-grounded analysis. Run from this file's directory or set DIINA_PROJECT_DIR.
find_project <- function() {
  candidates <- c(Sys.getenv("DIINA_PROJECT_DIR", ""), ".", "..", "../project_inputs", "project_inputs", "/mnt/data")
  candidates <- candidates[nzchar(candidates)]
  for (p in candidates) if (file.exists(file.path(p, "raw_results.csv"))) return(normalizePath(p))
  stop("Could not find raw_results.csv; set DIINA_PROJECT_DIR to project_inputs or source directory.")
}
project_dir <- find_project()
read_telemetry <- function() read.csv(file.path(project_dir, "raw_results.csv"), stringsAsFactors=FALSE)
# Audit CSV schema, seed membership, 300 records/seed, missingness, and value bounds.
d <- read_telemetry()
required <- c("seed","step","loss","fast_usage","slow_usage","entropy")
stopifnot(identical(names(d), required))
observed <- sort(unique(d$seed)); intended <- c(42,123,456,789,2022)
cat("Rows:",nrow(d)," columns:",paste(names(d),collapse=", "),"\n")
cat("Observed seeds:",paste(observed,collapse=", "),"\n")
cat("Requested seed set:",paste(intended,collapse=", "),"\n")
print(table(d$seed))
cat("Missing values by column:\n"); print(colSums(is.na(d)))
counts <- aggregate(step ~ seed, d, function(x) c(n=length(x), unique=length(unique(x)), min=min(x), max=max(x)))
print(counts)
stopifnot(nrow(d)==1500L, all(table(d$seed)==300L), all(!is.na(d)))
stopifnot(all(d$step %in% 1:300), all(d$loss>=0), all(d$fast_usage>=0 & d$fast_usage<=1), all(d$slow_usage>=0 & d$slow_usage<=1), all(d$entropy>=0))
stopifnot(all(abs(d$fast_usage+d$slow_usage-1)<1e-5))
cat("PASS: structural/range checks. Seed-set difference:",paste(setdiff(observed,intended),collapse=","),"; missing requested:",paste(setdiff(intended,observed),collapse=","),"\n")
