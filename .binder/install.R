# install.R
options(repos = c(CRAN = "https://cloud.r-project.org"))

r_packages <- c(
  "lme4",
  "pbkrtest",
  "car",
  "afex",
  "emmeans",
  "performance"
)

install.packages(r_packages)
