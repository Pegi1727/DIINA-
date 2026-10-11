# DIINA project-grounded analysis. Run from this file's directory or set DIINA_PROJECT_DIR.
find_project <- function() {
  candidates <- c(Sys.getenv("DIINA_PROJECT_DIR", ""), ".", "..", "../project_inputs", "project_inputs", "/mnt/data")
  candidates <- candidates[nzchar(candidates)]
  for (p in candidates) if (file.exists(file.path(p, "raw_results.csv"))) return(normalizePath(p))
  stop("Could not find raw_results.csv; set DIINA_PROJECT_DIR to project_inputs or source directory.")
}
project_dir <- find_project()
read_telemetry <- function() read.csv(file.path(project_dir, "raw_results.csv"), stringsAsFactors=FALSE)
# Educational R port of v17's exact five-command x repetition-1:5 mapping.
# Mirrors loader-level sequence expansion; source loader file is empty in the supplied package.
set.seed(42); commands <- c("jump","walk","run","look","turn"); actions <- toupper(commands)
pairs <- do.call(rbind,lapply(seq_along(commands),function(i)data.frame(command=commands[i],action=actions[i],repeat_n=1:5)))
pairs$input <- vapply(seq_len(nrow(pairs)),function(i)paste(rep(pairs$command[i],pairs$repeat_n[i]),collapse=" "),"")
pairs$output <- vapply(seq_len(nrow(pairs)),function(i)paste(rep(pairs$action[i],pairs$repeat_n[i]),collapse=" "),"")
pairs <- pairs[sample(seq_len(nrow(pairs))),]; train_n <- floor(.8*nrow(pairs)); train <- pairs[seq_len(train_n),]; test <- pairs[(train_n+1):nrow(pairs),]
print(head(pairs,8)); cat("Pairs:",nrow(pairs)," train:",nrow(train)," test:",nrow(test),"\n")
stopifnot(nrow(pairs)==25,all(vapply(seq_len(25),function(i)length(strsplit(pairs$input[i]," ")[[1]])==pairs$repeat_n[i],TRUE)))
