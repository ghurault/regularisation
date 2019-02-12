# Initialisation ----------------------------------------------------------

rm(list=ls()) # Clear Workspace

set.seed(1)

library(MASS)
library(rethinking)
library(glmnet)
library(rstan)
library(ggplot2)
library(cowplot)

# Parallel computing Stan
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# OPTIONS (main are condition, corr and SNR)
N_train <- 100 # Number of observations for training
N_test <- 1000 # Number of observations for testing
D <- 80 # Number of dimensions/features
beta_max <- 10 # Maximum value for beta (X is normalised)
condition <- "3" # determines the patterns of beta
corr <- TRUE # multicollinearity
SNR <- 2 # 1,2,4 ;cf. sd(Y)/sigma

beta_pattern <- function(beta){
  # Lollipop chart the true values of beta
  #
  # Args:
  # beta: vector of beta values
  #
  # Returns:
  # Ggplot of beta values
  
  library(ggplot2)
  
  ggplot(data = data.frame(Index = 1:length(beta), Beta = beta), aes(x = Index, y = Beta)) +
    geom_point() +
    geom_segment(aes(xend = Index, yend = 0)) +
    scale_y_continuous(breaks = seq(0, 10, 2)) +
    theme_bw(base_size = 15)
}

plot_beta <- function(beta, residuals = FALSE, count = FALSE){
  # Scatterplot of the true betas vs estimated betas
  #
  # Args:
  # beta: dataframe including the true and estimated betas
  # residuals: whether to plot residuals or estimated betas
  # count: whether to have the size of the dots proportional to their count
  #
  # Returns:
  # Ggplot of true vs estimated betas
  
  library(ggplot2)
  library(reshape2)
  
  cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
  
  tmp <- melt(beta, id.vars = "True", variable.name = "Method", value.name = "Coefficient")
  if (!residuals){
    tmp$y <- tmp$Coefficient
  }else{
    tmp$y <- tmp$Coefficient - tmp$True
  }
  p <- ggplot(data = tmp, aes(x = True, y = y, colour = Method)) +
    geom_path() +
    scale_colour_manual(values = cbbPalette)
  
  if (count){
    p <- p + geom_count()
  }else{
    p <- p + geom_point(size = 2)
  }
  
  if (!residuals){
    p <- p + geom_abline(intercept = 0, slope = 1, linetype = "dashed") + labs(x = "Coefficient", y = "Estimate", colour="")
  }else{
    p <- p + geom_hline(yintercept = 0, linetype = "dashed") + labs(x = "Coefficient", y = "Residual", colour="")
  }
  
  p <- p + theme_bw(base_size = 15) + theme(legend.position = "top")
  
  return(p)
}

performance_chart <- function(df, ref){
  # Barchart of Mean Square Error (MSE) between the column of a dataframe df and a reference ref
  #
  # Args:
  # df: dataframe
  # ref: vector of length nrow(df)
  #
  # Returns:
  # Ggplot of the MSE
  
  library(ggplot2)
  
  tmp <- data.frame(Mean = apply(df, 2, function(x){mean((x - ref)^2)}),
                    SE = apply(df, 2, function(x){sd((x - ref)^2) / sqrt(length(x))}))
  tmp$Method <- rownames(tmp)
  rownames(tmp) <- NULL
  tmp$Method <- factor(tmp$Method, levels = tmp$Method[order(tmp$Mean, decreasing = TRUE)])
  
  # Confidence interval from bootstrapping
  # library(boot);bootres <- boot(data=pred$OLS, statistic=function(data,indices){sqrt(mean((data[indices]-Y_test[indices])^2))},R=1000) # cf. bootstrap estimate
  
  ysup <- with(tmp, max(Mean + SE)) # where to plot the text and limits of the plot
  
  ggplot(data = tmp, aes(x = Method, y = Mean, ymin = Mean - SE, ymax = Mean + SE)) +
    geom_bar(stat = "identity") + geom_errorbar() +
    geom_text(aes(y = (Mean + SE + ysup * 0.03), label = paste(signif(Mean, 3), "±", signif(SE, 2)))) +
    scale_y_continuous(expand = c(0, 0), limits = c(0, ysup * 1.1)) +
    labs(x = "", y = "MSE") +
    theme_classic(base_size = 15) + theme(axis.text.x = element_text(angle = 30, hjust = 1))
}

# Generate fake data (y = x * beta + epsilon) ------------------------------------------------------

N_tot <- N_train + N_test

if (condition == "1"){
  K <- 10 # Number of non-zero features
  beta_true <- c(rep(0, D - K - 1), seq(0, beta_max, length.out = K + 1)) # 0 0 0 0 ... 2 4 6 ...
}
if (condition == "2"){
  beta_true <- sort(rep(seq(0, beta_max, 2), length.out = D)) # 0 0 0 2 2 2 4 4 4 ....
  K <- sum(beta_true != 0)
}
if (condition == "3"){
  beta_true <- (0:(D - 1))^2
  beta_true <- beta_true / max(beta_true) * beta_max # increasing beta (more around 0)
  K <- sum(beta_true != 0)
}
beta <- data.frame(True = beta_true)

if (!corr){
  X_tot <- matrix(rnorm(N_tot * D), nrow = N_tot, ncol = D)
}else{
  diag <- rep(1, D) # eigenvalues
  R <- rlkjcorr(n = 1, K = D, eta = .01) # LKJ distribution for correlation matrix (rethinking package)
  
  Sigma <- R * tcrossprod(diag) # covariance matrix
  Mu <- rep(0, D)
  X_tot <- mvrnorm(n = N_tot, rep(0, D), Sigma) # Multivariate normal (MASS package)
  # library(corrplot);corrplot(cor(X_tot)[1:10, 1:10], method = "color") # Visualise correlation in X
}

# beta_pattern(beta_true) # Visualise the patterns of betas

X_train <- X_tot[1:N_train,]
X_test <- X_tot[(N_train + 1):N_tot,]

Y_tot <- X_tot %*% beta$True
sigma <- sd(Y_tot) / SNR
Y_tot <- as.numeric(Y_tot + rnorm(N_tot, sd = sigma))
Y_train <- Y_tot[1:N_train]
Y_test <- Y_tot[(N_train + 1):N_tot]

# OLS/Lasso/Ridge/Elastic Net -------------------------------------------------------------------

# OLS
ols <- glm.fit(X_train, Y_train)
beta$OLS <- ols$coefficients

lambda="lambda.1se"
alpha <- .5

# Lasso
lasso <- cv.glmnet(X_train, Y_train, nlambda = 100, alpha = 1, nfolds = 10)
beta$Lasso <- as.numeric(coef(lasso, s = lambda)[-1])

# Ridge
ridge <- cv.glmnet(X_train, Y_train, nlambda = 100, alpha = 0, nfolds=10)
beta$Ridge <- as.numeric(coef(ridge, s = lambda)[-1])

# Elastic Net
enet <- cv.glmnet(X_train, Y_train, nlambda=100, alpha = alpha, nfolds=10)
beta$ENET <- as.numeric(coef(enet, s = enet[[lambda]])[-1])
# Rescaled Elastic Net
beta$ENET_rescaled <- (1 + enet[[lambda]] * (1 - alpha) / 2) * beta$ENET

# Hybrid OLS-Lasso
lasso_ols <- glm.fit(X_train[, beta$Lasso > 0], Y_train)
beta$Lasso_OLS <- beta$Lasso
beta$Lasso_OLS[beta$Lasso > 0] <- lasso_ols$coefficients

# Relaxed Lasso
rel_lasso <- cv.glmnet(X_train[, beta$Lasso > 0], Y_train, nlambda = 100, alpha = 1,nfolds = 10)
beta$Relaxed_Lasso <- beta$Lasso
beta$Relaxed_Lasso[beta$Lasso > 0] <- as.numeric(coef(rel_lasso, s = "lambda.1se")[-1])

# Regularised Horseshoe ---------------------------------------------------

p0 <- K # prior guess on number of non-zero features: oracle (true) guess is K but fine if K/2 or 2*K for instance

data_stan <- list(n = N_train,
                  d = D,
                  y = Y_train,
                  x = X_train,
                  scale_icept = 1,
                  scale_global = p0 / (D - p0) / sqrt(N_train), # cf. prior guess on number of non-zero features
                  nu_global = 1, # cf. prior for tau: 1=cauchy recommended distribution by J. Piironen and A. Vehtari
                  nu_local = 1, # cf. prior for lambda, cf. horseshoe
                  slab_scale = 2, # cf. scale for non-zero parameters
                  slab_df = 1) # cf. distribution for non-zero parameters 1=cauchy

param <- c("sigma", "beta0", "beta", "tau", "lambda", "c", "y_rep")

## Fit
fit <- stan(file = "regularised_horseshoe.stan", data = data_stan, iter = 4000, chains = 6, pars = param, control = list(adapt_delta = .995))

## Diagnostics
# pairs(fit, pars = c("sigma", "beta0", "tau", "c"))
# print(fit)
# library(shinystan);launch_shinystan(fit) # cf. PPC

## Betas
s <- summary(fit, pars = "beta", probs = c(.025, .05, .5, .95, .975))$summary
beta$Regularised_Horseshoe <- s[, "mean"]
beta$Regularised_Horseshoe_Lower <- s[, "2.5%"]
beta$Regularised_Horseshoe_Upper <- s[, "97.5%"]
# beta$Regularised_Horseshoe_Sig <- s[, "mean"] * ((0 < s[, "5%"]) | (0 > s[, "95%"])) # Apply a cut-off: set to zero if 90% CI include 0
beta$Regularised_Horseshoe_NZ <- s[, "mean"] * (abs(s[, "mean"]) > sd(Y_train) * .05) # >1, . sd(Y_train)*.1

## Coefficient plot with CI
# plot(fit, pars = "beta")
# ggplot(data = beta,
#        aes(x = True, y = Regularised_Horseshoe, ymin = Regularised_Horseshoe_Lower, ymax = Regularised_Horseshoe_Upper)) +
#   geom_pointrange() +
#   geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
#   labs(x = "True coefficient", y = "Estimated coefficient") +
#   theme_bw(base_size = 15)

# saveRDS(s, file = "beta_cond1_snr2_corr_over.rds")

# Compare methods ---------------------------------------------------------------

# Coefficients plots
pl1 <- lapply(colnames(beta)[!(colnames(beta) %in% c("True", "Regularised_Horseshoe_Lower", "Regularised_Horseshoe_Upper"))],
              function(x){
                plot_beta(beta[, c("True", x)], count = FALSE) +
                  # labs(subtitle = paste("RMSE =", signif(sqrt(mean((beta$True - beta[, x])^2)), 3))) +
                  labs(title = x) + theme(legend.position = "none")
              })
plot_grid(plotlist = pl1, ncol = 3)

# plot_beta(beta[, c("True", "OLS")])
# plot_beta(beta[, c("True", "OLS", "Lasso")], count = TRUE)
# plot_beta(beta)

# MSE coefficients (might need to remove OLS or ENET_rescaled to see something)
performance_chart(beta[, !(colnames(beta) %in% c("True", "OLS", "Regularised_Horseshoe_Lower", "Regularised_Horseshoe_Upper"))], beta$True) +
  # scale_y_log10() +
  scale_y_sqrt() +
  labs(title = "Coefficients")

# Prediction
pred <- cbind(data.frame(True = Y_test),
              sapply(colnames(beta)[!(colnames(beta) %in% c("True", "Regularised_Horseshoe_Lower", "Regularised_Horseshoe_Upper"))],
                     function(x){X_test %*% beta[, x]}))
# For horseshoe, possibility to integrate over parameter uncertainty (full Bayesian, need to implement this in Stan)

# MSE predictions (red line for noise level)
performance_chart(pred[, colnames(pred) != "True"], Y_test) +
  geom_hline(yintercept = sigma^2, colour = "red") +
  # scale_y_log10() +
  scale_y_sqrt() +
  labs(title = "Predictions")

# Other plots -------------------------------------------------------------

## Prediction plots
pl2 <- lapply(colnames(pred)[-1],
              function(x){
                tmp <- as.data.frame(pred[, c("True", x)])
                colnames(tmp) <- c("Actual", "Prediction")
                
                ggplot(data = tmp, aes(x = Actual, y = Prediction)) +
                  geom_point() +
                  geom_abline(slope = 1, intercept = 0) +
                  labs(title = paste(x)) +
                  theme_bw(base_size = 15)
              })
plot_grid(plotlist = pl2, ncol = 3)

## Residual plot (density)
tmp <- apply(pred, 2, function(x){x - pred$True})[, -1]
tmp <- melt(tmp, varnames = c("ID", "Method"), value.name = "Residual")
# tmp <- subset(tmp, Method %in% c("Regularised_Horseshoe", "Lasso_OLS", "Relaxed_Lasso"))
ggplot(data = tmp, aes(x = Residual, fill = Method)) + geom_density(alpha = .3)

## p0-dependence
s <- readRDS("beta_cond1_snr2_corr_oracle.rds")
tmp <- data.frame(True = beta_true, Mean_Estimate = s[, "mean"], Lower_Estimate = s[, "2.5%"], Upper_Estimate = s[, "97.5%"], Label = "Oracle")
s <- readRDS("beta_cond1_snr2_corr_under.rds")
tmp <- rbind(tmp, data.frame(True = beta_true, Mean_Estimate = s[, "mean"], Lower_Estimate = s[, "2.5%"], Upper_Estimate = s[, "97.5%"], Label = "Underestimating"))
s <- readRDS("beta_cond1_snr2_corr_over.rds")
tmp <- rbind(tmp, data.frame(True = beta_true, Mean_Estimate = s[, "mean"], Lower_Estimate = s[, "2.5%"], Upper_Estimate = s[, "97.5%"], Label = "Overestimating"))

tmp$Index <- 1:nrow(s)
tmp$Label <- factor(tmp$Label, levels = c("Underestimating", "Oracle", "Overestimating"))
tmp <- tmp[order(tmp$Index, tmp$Label),]

pos=1:(4 * nrow(s))
tmp$Position=pos[pos %% 4 > 0] # add spacing between betas

ggplot(data = tmp, aes(x = Position, colour = Label)) +
  geom_pointrange(aes(y = Mean_Estimate, ymin = Lower_Estimate, ymax = Upper_Estimate), size=1.1) +
  scale_x_continuous(breaks = seq(2, length(pos), 4), labels = seq(1:nrow(s))) +
  labs(x = "Index", y = "Estimate", colour = "") +
  theme_classic(base_size = 15)+theme(axis.text.x = element_text(angle = 90), legend.position = "top")
