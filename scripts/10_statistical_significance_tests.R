# DIINA project-grounded analysis. Run from this file's directory or set DIINA_PROJECT_DIR.
find_project <- function() {
  candidates <- c(Sys.getenv("DIINA_PROJECT_DIR", ""), ".", "..", "../project_inputs", "project_inputs", "/mnt/data")
  candidates <- candidates[nzchar(candidates)]
  for (p in candidates) if (file.exists(file.path(p, "raw_results.csv"))) return(normalizePath(p))
  stop("Could not find raw_results.csv; set DIINA_PROJECT_DIR to project_inputs or source directory.")
}
project_dir <- find_project()
read_telemetry <- function() read.csv(file.path(project_dir, "raw_results.csv"), stringsAsFactors=FALSE)
# Paired seed-level convergence comparisons: initial vs final loss; tiny n=5, interpret cautiously.
d<-read_telemetry(); seeds<-sort(unique(d$seed)); stopifnot(length(seeds)==5)
wide<-do.call(rbind,lapply(split(d,d$seed),function(x){x<-x[order(x$step),];data.frame(seed=x$seed[1],initial=x$loss[1],final=x$loss[nrow(x)],mean_loss=mean(x$loss))}))
print(wide); delta<-wide$initial-wide$final; cat("Mean paired reduction:",mean(delta),"\n")
print(t.test(wide$initial,wide$final,paired=TRUE)); print(wilcox.test(wide$initial,wide$final,paired=TRUE,exact=FALSE))
# Repeated-measures ANOVA on pre-specified 3 windows (steps 1-100,101-200,201-300); seed is subject.
d$window<-cut(d$step,breaks=c(0,100,200,300),labels=c("early","middle","late")); means<-aggregate(loss~seed+window,d,mean); print(means)
if(requireNamespace("stats",quietly=TRUE)) print(summary(aov(loss~window+Error(factor(seed)/window),data=means)))
cat("Caution: five seeds provide low power; telemetry steps are not independent replicates.\n")
