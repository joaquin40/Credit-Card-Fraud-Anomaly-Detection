## ---------------------------------------------------------------------
pacman::p_load(tidyverse, data.table, DataExplorer, plotly, skimr, cowplot, caret, isotree, randomForest)


## ---------------------------------------------------------------------
df <- fread("./data/card_transdata.csv")


## ---------------------------------------------------------------------
glimpse(df)

## ---------------------------------------------------------------------
table(df$fraud) |> prop.table()


## ---------------------------------------------------------------------
DataExplorer::plot_missing(df)
dev.copy(png, "./images/na.png")
dev.off()
skim(df)


## ---------------------------------------------------------------------
gg_online <- df |> 
  ggplot(aes(factor(online_order), fill = as.factor(fraud) )) + 
  geom_bar(position = "fill") + 
  scale_fill_discrete(labels = c("No", "Yes")) +
  scale_x_discrete(labels = c("No", "Yes"))+
  labs(x = "Online Order", fill = "Fraud", y = "Proportional") 

gg_pin <- df |> 
  ggplot(aes(factor(used_pin_number), fill = as.factor(fraud) )) + 
  geom_bar(position = "fill") + 
  scale_fill_discrete(labels = c("No", "Yes")) +
  scale_x_discrete(labels = c("No", "Yes"))+
  labs(x = "Used Pin Number", fill = "Fraud", y = "Proportional") 

ggsave("./images/fraud_eda.png",plot = cowplot::plot_grid(gg_online, gg_pin, ncol = 1))

df |> 
  ggplot(aes(distance_from_home, distance_from_last_transaction , color = as.factor(fraud) )) + 
  geom_point() + 
  scale_color_discrete(labels = c("No", "Yes")) +
  labs(x = "Distance from home", color = "Fraud", y = "Distance from last transaction") 


## ---------------------------------------------------------------------
pr_out <- prcomp(df, scale = TRUE)
summary(pr_out)


## ---------------------------------------------------------------------
pve <- summary(pr_out)$importance[2,]


## ---------------------------------------------------------------------
pve_gg <- ggplot(data.frame(x = seq.int(pve), pve = pve ) , aes(x, pve))  + 
  geom_line() + 
  geom_point(color = "blue", size = 2) +
  scale_x_continuous(breaks = seq.int(pve)) + 
  labs(x  = "",
      y = "Proportional \n Variance Explained")

sum_pve_gg <- ggplot(data.frame(x = seq.int(pve), pve = cumsum(pve) ) , aes(x, pve))  + 
  geom_line() + 
   geom_point(color = "blue", size = 2) + 
  scale_x_continuous(breaks = seq.int(pve)) + 
  labs(x  = "Principle component",
      y = "Cumulative Proportional \n Variance Explained")


ggsave("./images/pca_var.png",plot = cowplot::plot_grid(pve_gg, sum_pve_gg, ncol = 1))



## ---------------------------------------------------------------------
cowplot::plot_grid(pve_gg, sum_pve_gg, ncol = 1)


## ---------------------------------------------------------------------
library(factoextra)
fviz_pca_biplot(pr_out)
dev.copy(png, "./images/biplot.png")
dev.off()


## ---------------------------------------------------------------------
df1 <- df |> 
  mutate(fraud = as.factor(fraud))


## ---------------------------------------------------------------------
set.seed(1)
index <- createDataPartition(df1$fraud, p = .7, list = FALSE)
train <- df1[index,] 
test <- df1[-index,]

train_x <- df1[index,] |> select(-fraud)
train_y <- df1[index,] |> select(fraud)
test_x <- df1[-index,]  |> select(-fraud)
test_y <- df1[-index,]  |> select(fraud)


## ---------------------------------------------------------------------
prop.table(table(train$fraud))
prop.table(table(test$fraud))


## ---------------------------------------------------------------------

model <-  isotree::isolation.forest(train_x,
                                    ntree = 500,
                                    nthreads = 10,
                                    sample_size = 256,  
                                    missing_action="fail")

# isolation depth number of partitions that it takes to isolate a point
#avg_depth <- predict(model, test, type = "avg_depth")

pred <-  predict(model, test_x)




## ---------------------------------------------------------------------
library(randomForest)


method = c("rf")
tune <-  expand.grid(mtry = 1:4)

ctrl <- trainControl(method = "cv", number = 10)

rf <- train(fraud ~. , 
             data = train,
             method = method,
             trControl = ctrl,
             tuneGrid = tune)


pred_rf <- predict(rf, newdata = test, type = "prob")


## ---------------------------------------------------------------------
library(pROC)

roc <- roc(test_y$fraud, pred)
auc <- auc(roc)

roc_df <- data.frame("spec1" = 1 - roc$specificities, sens = roc$sensitivities)

roc_rf <- roc(test$fraud, pred_rf[,2])
auc_rf <- auc(roc_rf)



## ---------------------------------------------------------------------
plot(roc_df$spec1, roc_df$sens, type = "l", main = "ROC",
     xlab =  "1 - Specificity", ylab = "Sensitivity", col = "#FC4E07", lwd = 3)
lines(1 - roc_rf$specificities, roc_rf$sensitivities, col = "#00AFBB" , lwd = 3)
abline(a = c(0,0), b = c(1,1), lwd = 3)
axis(side = 2, tcl = 0)
# Force y-axis ticks to show
axis(side = 2, at = seq(0.2, .8,by = .2), labels = seq(0.2, .8,by = .2))
text(x = 0.60, y = .18, labels = paste("AUC:", round(auc,3)))
text(x = 0.60, y = .08, labels = paste("AUC:", round(auc_rf,3)))
legend("bottomright", legend = c("Isolation Forest", "Random Forest"),
       col = c("#FC4E07", "#00AFBB"), lwd = 2, bty = "n")

dev.copy(png, "./images/roc.png")
dev.off()


## ---------------------------------------------------------------------
pred_rf_val <- predict(rf, newdata = test)


## ---------------------------------------------------------------------
confusionMatrix(test$fraud, reference = pred_rf_val, positive = "1", mode = "prec_recall")


## ---------------------------------------------------------------------
varImp(rf)


## ---------------------------------------------------------------------
varImpPlot(rf$finalModel)
dev.copy(png, "./images/varImpPlot.png")
dev.off()


## ---------------------------------------------------------------------
importance(rf$finalModel) |> 
  data.frame() |> 
  rownames_to_column(var = "Variables") |> 
  gt::gt()


