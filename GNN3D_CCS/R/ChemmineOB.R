if (!requireNamespace("ChemmineOB", quietly = TRUE)){
  if (!requireNamespace("BiocManager", quietly = TRUE)){
    install.packages("BiocManager")
  }
  BiocManager::install("ChemmineOB")
}

OB_convertFormat <- function(from, to, source){
  output <- ChemmineOB::convertFormat(from, to, source, options=data.frame(names="gen3D",args=""))
  return (output)
}