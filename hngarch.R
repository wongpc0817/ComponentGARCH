library('fOptions')
source('CmpntHngarch.R')
source('new_hngarch.R')
source('UdHngarch.R')
source('CmpntHngarchOption.R')

df <- read.csv("data/SPX/daily_spx.csv")
ts <- diff(log(df$Close))
# Calculate z-scores
z_scores <- (ts - mean(ts)) / sd(ts)
# 
# # Define a threshold (e.g., 3 standard deviations)
threshold <- 3
# 
# # Remove outliers (values beyond threshold)
ts <- ts[abs(z_scores) <= threshold]

# If you want to replace outliers with NA
ts[abs(z_scores) > threshold] <- NA


model = list(lambda = 4, omega = 8e-8, alpha = 6e-8,
             beta = 0.7, gamma = 0.1, rf = 0.1)
# model = list(lambda = 4, sigma2 = 8e-8, alpha_s = 6e-8,
#              beta_s = 0.7, gamma_s = 0.01 , alpha_q = 6e-8,
#              beta_q = 0.7, gamma_q = 0.01 , rf = 0.01)
# ts = hngarchSim(model = model, n = 500, n.start = 100)
# par(mfrow = c(2, 1), cex = 0.75)
# ts.plot(ts, col = "steelblue", main = "HN Garch Symmetric Model")
# 
# grid()
mle <- hngarchFit(model = model, x=ts, symmetric = FALSE)
# mle <- new_hngarchFit(model = model, x=ts, symmetric = FALSE)
## hngarchFit - 
# HN-GARCH log likelihood Parameter Estimation:
# To speed up, we start with the simulated model ...
# mle = CmpntHngarchFit(model = model, x = ts, symmetric = FALSE)
mle
# pred <- CmpntHngarchSim(mle$model)
pred <- hngarchSim(mle$model, n=length(ts))

plot(pred,type ="l")

print(shapiro.test(pred - ts))


# lines(mle$x, col = rgb(1, 0, 0, 0.5))
# futures <- CmpntHNGOption(mle$model,10,252)
# print(futures)

## summary.hngarch - 
# HN-GARCH Diagnostic Analysis:
# par(mfrow = c(3, 1), cex = 0.75)
# summary(mle)

## hngarchStats - 
# HN-GARCH Moments:
# hngarchStats(mle$model)