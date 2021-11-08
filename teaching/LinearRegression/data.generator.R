generate.data <- function(n=10000, noise.factor=0.1) {
  age <- rnorm(n, 35, 5)
  sex <- sample(c('m', 'f'), n, replace=T)
  weight <- 130 + rt(n, 7)
  n.male <- length(weight[sex == 'm'])
  weight[sex == 'm'] <- weight[sex == 'm'] + rnorm(n.male, 20, 8)
  height <- rnorm(n, 110, 12)
  height[sex == 'm'] <- height[sex == 'm'] + rnorm(n.male, 45, 15)
  mins.exercise <- 6 * rchisq(n, 5)	
  g.carb <- rnorm(n, 1400, 180) 
  g.fat <- rnorm(n, 800, 100)
  g.protein <- rnorm(n, 1000, 145)
  day.of.year <- round(runif(n, 1, 365))
  hours.sleep <- rnorm(n, 50, 2)
  cortisol <- 100 * rpois(n, 4)
  y <- (
    500000
    + 1.2*log(age) 
    + 20*(sex == 'm') 
    + 2.7*weight 
    - 0.2*height 
    + 1.5*mins.exercise 
    - 0.8*(mins.exercise)^1.2 
    + 10*sin(2*pi*(day.of.year + 150) / 365) 
    + 3.3*sqrt(cortisol)*(cortisol > 500) 
    + 0.77*cortisol*(cortisol <= 500)
    - 0.6*log(hours.sleep)*(hours.sleep >= 50) 
    + 0.2*(56 - hours.sleep)*(hours.sleep < 56)
    + 0.02*mins.exercise*weight 
    - 0.01*g.protein*height*weight)
  
  rng <- max(y) - min(y)
  noise <- rnorm(n, sd=noise.factor * rng)
  y <- y + noise
  data.frame(y, age, sex, weight, height, mins.exercise, g.carb, g.fat, g.protein, 
             day.of.year, hours.sleep, cortisol)
}

