# DIINA project-grounded analysis. Run from this file's directory or set DIINA_PROJECT_DIR.
find_project <- function() {
  candidates <- c(Sys.getenv("DIINA_PROJECT_DIR", ""), ".", "..", "../project_inputs", "project_inputs", "/mnt/data")
  candidates <- candidates[nzchar(candidates)]
  for (p in candidates) if (file.exists(file.path(p, "raw_results.csv"))) return(normalizePath(p))
  stop("Could not find raw_results.csv; set DIINA_PROJECT_DIR to project_inputs or source directory.")
}
project_dir <- find_project()
read_telemetry <- function() read.csv(file.path(project_dir, "raw_results.csv"), stringsAsFactors=FALSE)
# v19.1 exact sequence pattern: one random key + distance random noise tokens + QUERY.
set.seed(42); keys <- c("A","B","C","D"); noise_tokens <- c("x","y","z","n"); distances <- c(50,100,200)
make_sequence <- function(distance){key<-sample(keys,1); distractors<-sample(noise_tokens,distance,replace=TRUE); list(sequence=c(key,distractors,"QUERY"),target=key)}
complexity <- do.call(rbind,lapply(distances,function(d){z<-make_sequence(d); data.frame(distance=d,sequence_length=length(z$sequence),distractor_count=sum(z$sequence %in% noise_tokens),target=z$target)}))
print(complexity); stopifnot(all(complexity$sequence_length==distances+2),all(complexity$distractor_count==distances))
cat("v19.1 uses 300 train and 100 test examples per distance; v19.0 uses 800/300 and additionally distance 500.\n")
