# DIINA project-grounded analysis. Run from this file's directory or set DIINA_PROJECT_DIR.
find_project <- function() {
  candidates <- c(Sys.getenv("DIINA_PROJECT_DIR", ""), ".", "..", "../project_inputs", "project_inputs", "/mnt/data")
  candidates <- candidates[nzchar(candidates)]
  for (p in candidates) if (file.exists(file.path(p, "raw_results.csv"))) return(normalizePath(p))
  stop("Could not find raw_results.csv; set DIINA_PROJECT_DIR to project_inputs or source directory.")
}
project_dir <- find_project()
read_telemetry <- function() read.csv(file.path(project_dir, "raw_results.csv"), stringsAsFactors=FALSE)
# Reconcile observed files to claims, without treating manuscript prose as measured telemetry.
if (!requireNamespace("jsonlite",quietly=TRUE)) stop("Install jsonlite")
d<-read_telemetry(); s<-jsonlite::fromJSON(file.path(project_dir,"summary.json")); one<-jsonlite::fromJSON(file.path(project_dir,"result_seed_2022.json")); read_text<-function(f)paste(readLines(file.path(project_dir,f),warn=FALSE),collapse="\n")
readme<-read_text("README.md"); paper<-read_text("NeurIPS_Camera_Ready_Final.txt"); manuscript_note<-"manuscript.docx: dual-stream Fast/Slow theory and Linguistic Inhibition Gate (LIG); benchmark code is simplified gated-GRU proxy, not that full architecture."
seed_mean<-aggregate(loss~seed,d,mean); final<-do.call(rbind,lapply(split(d,d$seed),function(x)data.frame(seed=x$seed[1],final=x$loss[which.max(x$step)])))
cat(manuscript_note,"\nObserved seeds:",paste(sort(unique(d$seed)),collapse=", "),"\n")
cat("CSV pooled mean loss:",mean(d$loss),"; summary avg_loss mean:",s$avg_loss$mean,"; seed-2022 JSON avg/final:",one$avg_loss,one$final_loss,"\n")
cat("Seed-2022 CSV avg/final:",mean(d$loss[d$seed==2022]),tail(d$loss[d$seed==2022],1),"\n")
cat("README mentions seed 1011:",grepl("1011",readme),"; CSV includes 1011:",1011 %in% d$seed,"; README claims 456:",grepl("456",readme),"; CSV includes 456:",456 %in% d$seed,"\n")
cat("Reported claims are text-only unless backed by available CSV/JSON; manuscript and benchmarking framework differ in scope.\n")
