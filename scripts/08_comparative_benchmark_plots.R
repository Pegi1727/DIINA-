# DIINA project-grounded analysis. Run from this file's directory or set DIINA_PROJECT_DIR.
find_project <- function() {
  candidates <- c(Sys.getenv("DIINA_PROJECT_DIR", ""), ".", "..", "../project_inputs", "project_inputs", "/mnt/data")
  candidates <- candidates[nzchar(candidates)]
  for (p in candidates) if (file.exists(file.path(p, "raw_results.csv"))) return(normalizePath(p))
  stop("Could not find raw_results.csv; set DIINA_PROJECT_DIR to project_inputs or source directory.")
}
project_dir <- find_project()
read_telemetry <- function() read.csv(file.path(project_dir, "raw_results.csv"), stringsAsFactors=FALSE)
# Publication-style telemetry panels with across-seed mean +/- 95% t confidence ribbons.
if (!requireNamespace("ggplot2",quietly=TRUE)) stop("Install ggplot2")
d <- read_telemetry(); long <- do.call(rbind,lapply(c("loss","fast_usage","slow_usage","entropy"),function(v)data.frame(seed=d$seed,step=d$step,metric=v,value=d[[v]])))
agg <- aggregate(value~metric+step,long,function(x)c(mean=mean(x),sd=sd(x),n=length(x)))
agg$mean <- agg$value[,"mean"]; agg$sd <- agg$value[,"sd"]; agg$n <- agg$value[,"n"]; agg$ci <- qt(.975,agg$n-1)*agg$sd/sqrt(agg$n)
p <- ggplot2::ggplot(agg,ggplot2::aes(step,mean,color=metric,fill=metric))+ggplot2::geom_ribbon(ggplot2::aes(ymin=mean-ci,ymax=mean+ci),alpha=.16,color=NA)+ggplot2::geom_line(linewidth=.65)+ggplot2::facet_wrap(~metric,scales="free_y",ncol=2)+ggplot2::labs(title="DIINA telemetry across seeds",subtitle="Mean and pointwise 95% t confidence intervals",x="Training step",y="Metric")+ggplot2::theme_minimal(base_size=12)+ggplot2::theme(legend.position="none")
print(p); out <- file.path(project_dir,"diina_telemetry_panels.png"); ggplot2::ggsave(out,p,width=9,height=6,dpi=300); cat("Saved:",out,"\n")
