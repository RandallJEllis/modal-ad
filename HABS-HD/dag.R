ggdag::ggdag(dag, text = FALSE) +
  ggplot2::theme(legend.position = "none")
library(dagitty)

dag <- dagitty('dag {
  "Confounders" [pos="0,-0.1"]                            
  "pTau-217" [exposure, pos="-1,0"]
  AD [outcome, pos="1,0"]

  "Confounders" -> "pTau-217"
  "Confounders" -> AD
  "pTau-217" -> AD
}')


plot(dag)
  
  