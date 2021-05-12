library(ggplot2)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(cowplot)


setwd('/Users/eli/Documents/Spring 2021/CS8321/CS83212-Advanced-Neural-Networks/Final-Project/results')

my_theme = theme_cowplot(font_size=7) +
  theme(text = element_text(size=7),
        axis.text = element_text(size=7),
        axis.title = element_text(size=8), 
        legend.title = element_blank(),
        legend.text = element_text(size = 7),
        legend.key.size = unit(0.2, "cm"),
        line = element_line(size= 0.1),
        strip.text = element_text(margin = margin(0.1,0.1,0.1,0.1,'cm'), vjust =1, hjust=0.5),
        plot.margin = unit(c(0.05,0.05,0.05,0.05),'cm'),
        plot.title = element_text(size = 10, hjust = 0.5))

#-------- load data ---------

#classes
# 0 - airplane
# 1 - automobile
# 2 - bird
# 3 - cat
# 4 - deer 
# 5 - dog
# 6 - frog
# 7 - horse
# 8 - ship
# 9 - truck

col.names <- c('epoch', 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#baseline
baseline.loss.1 <- read.csv('./baseline/baseline_loss_1.csv', col.names = col.names)
baseline.loss.2 <- read.csv('./baseline/baseline_loss_2.csv', col.names = col.names)
baseline.loss.3 <- read.csv('./baseline/baseline_loss_3.csv', col.names = col.names)
baseline.acc.1 <- read.csv('./baseline/baseline_acc_1.csv', col.names = col.names)
baseline.acc.2 <- read.csv('./baseline/baseline_acc_2.csv', col.names = col.names)
baseline.acc.3 <- read.csv('./baseline/baseline_acc_3.csv', col.names = col.names)

baseline.train.1 <- read.csv('./baseline/baseline_trainloss_1.csv', col.names = c("epoch", "acc", "val_loss"))
baseline.train.2 <- read.csv('./baseline/baseline_trainloss_2.csv', col.names = c("epoch", "acc", "val_loss"))
baseline.train.3 <- read.csv('./baseline/baseline_trainloss_3.csv', col.names = c("epoch", "acc", "val_loss"))

#penalty 1e-2
penalty.001.loss.1 <- read.csv('./penalty_1e-2/penalty_001_loss_1.csv', col.names = col.names)
penalty.001.loss.2 <- read.csv('./penalty_1e-2/penalty_001_loss_2.csv', col.names = col.names)
penalty.001.loss.3 <- read.csv('./penalty_1e-2/penalty_001_loss_3.csv', col.names = col.names)
penalty.001.acc.1 <- read.csv('./penalty_1e-2/penalty_001_acc_1.csv', col.names = col.names)
penalty.001.acc.2 <- read.csv('./penalty_1e-2/penalty_001_acc_2.csv', col.names = col.names)
penalty.001.acc.3 <- read.csv('./penalty_1e-2/penalty_001_acc_3.csv', col.names = col.names)

penalty.001.train.1 <- read.csv('./penalty_1e-2/penalty_001_trainloss_1.csv', col.names = c("epoch", "acc", "val_loss"))
penalty.001.train.2 <- read.csv('./penalty_1e-2/penalty_001_trainloss_2.csv', col.names = c("epoch", "acc", "val_loss"))
penalty.001.train.3 <- read.csv('./penalty_1e-2/penalty_001_trainloss_3.csv', col.names = c("epoch", "acc", "val_loss"))

#penalty 0.5
penalty.0.5.loss.1 <- read.csv('./penalty_5e-1/penalty_05_loss_1.csv', col.names = col.names)
penalty.0.5.loss.2 <- read.csv('./penalty_5e-1/penalty_05_loss_2.csv', col.names = col.names)
penalty.0.5.loss.3 <- read.csv('./penalty_5e-1/penalty_05_loss_3.csv', col.names = col.names)
penalty.0.5.acc.1 <- read.csv('./penalty_5e-1/penalty_05_acc_1.csv', col.names = col.names)
penalty.0.5.acc.2 <- read.csv('./penalty_5e-1/penalty_05_acc_2.csv', col.names = col.names)
penalty.0.5.acc.3 <- read.csv('./penalty_5e-1/penalty_05_acc_3.csv', col.names = col.names)

penalty.0.5.train.1 <- read.csv('./penalty_5e-1/penalty_05_trainloss_1.csv', col.names = c("epoch", "acc", "val_loss"))
penalty.0.5.train.2 <- read.csv('./penalty_5e-1/penalty_05_trainloss_2.csv', col.names = c("epoch", "acc", "val_loss"))
penalty.0.5.train.3 <- read.csv('./penalty_5e-1/penalty_05_trainloss_3.csv', col.names = c("epoch", "acc", "val_loss"))

#penalty 1e-2 reversed
penalty.001.r.loss.1 <- read.csv('penalty_1e-2_rev/penalty_001_r_loss_1.csv', col.names = col.names)
penalty.001.r.loss.2 <- read.csv('penalty_1e-2_rev/penalty_001_r_loss_2.csv', col.names = col.names)
penalty.001.r.loss.3 <- read.csv('penalty_1e-2_rev/penalty_001_r_loss_3.csv', col.names = col.names)
penalty.001.r.acc.1 <- read.csv('penalty_1e-2_rev/penalty_001_r_acc_1.csv', col.names = col.names)
penalty.001.r.acc.2 <- read.csv('penalty_1e-2_rev/penalty_001_r_acc_2.csv', col.names = col.names)
penalty.001.r.acc.3 <- read.csv('penalty_1e-2_rev/penalty_001_r_acc_3.csv', col.names = col.names)

penalty.001.r.train.1 <- read.csv('./penalty_1e-2_rev/penalty_001_r_trainloss_1.csv', col.names = c("epoch", "acc", "val_loss"))
penalty.001.r.train.2 <- read.csv('./penalty_1e-2_rev/penalty_001_r_trainloss_2.csv', col.names = c("epoch", "acc", "val_loss"))
penalty.001.r.train.3 <- read.csv('./penalty_1e-2_rev/penalty_001_r_trainloss_3.csv', col.names = c("epoch", "acc", "val_loss"))

# ---------- pivot and join baseline data -------- 
baseline.pivot.loss.1 <- baseline.loss.1 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '1', type='baseline')

baseline.pivot.loss.2 <- baseline.loss.2 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '2', type='baseline')

baseline.pivot.loss.3 <- baseline.loss.3 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '3', type='baseline')

baseline <-  rbind(
    baseline.pivot.loss.1,
    baseline.pivot.loss.2,
    baseline.pivot.loss.3
  )

baseline.pivot.acc.1 <- baseline.acc.1 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '1', type='baseline')

baseline.pivot.acc.2 <- baseline.acc.2 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '2', type='baseline')

baseline.pivot.acc.3 <- baseline.acc.3 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '3', type='baseline')

baseline.acc <-  rbind(
  baseline.pivot.acc.1,
  baseline.pivot.acc.2,
  baseline.pivot.acc.3
)

baseline <- baseline %>% left_join(
  baseline.acc
)


# ---------- pivot and join penalty 1e-2 data ----------
penalty.001.pivot.loss.1 <- penalty.001.loss.1 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '1', type='1e-2')

penalty.001.pivot.loss.2 <- penalty.001.loss.2 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '2', type='1e-2')

penalty.001.pivot.loss.3 <- penalty.001.loss.3 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '3', type='1e-2')

penalty.001 <-  rbind(
  penalty.001.pivot.loss.1,
  penalty.001.pivot.loss.2,
  penalty.001.pivot.loss.3
)

penalty.001.pivot.acc.1 <- penalty.001.acc.1 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '1', type='1e-2')

penalty.001.pivot.acc.2 <- penalty.001.acc.2 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '2', type='1e-2')

penalty.001.pivot.acc.3 <- penalty.001.acc.3 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '3', type='1e-2')

penalty.001.acc <-  rbind(
  penalty.001.pivot.acc.1,
  penalty.001.pivot.acc.2,
  penalty.001.pivot.acc.3
)

penalty.001 <- penalty.001 %>% left_join(
  penalty.001.acc
)







































# ---------- pivot and join penalty 0.5 data ----------
penalty.0.5.pivot.loss.1 <- penalty.0.5.loss.1 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '1', type='0.5')

penalty.0.5.pivot.loss.2 <- penalty.0.5.loss.2 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '2', type='0.5')

penalty.0.5.pivot.loss.3 <- penalty.0.5.loss.3 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '3', type='0.5')

penalty.0.5 <-  rbind(
  penalty.0.5.pivot.loss.1,
  penalty.0.5.pivot.loss.2,
  penalty.0.5.pivot.loss.3
)

penalty.0.5.pivot.acc.1 <- penalty.0.5.acc.1 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '1', type='0.5')

penalty.0.5.pivot.acc.2 <- penalty.0.5.acc.2 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '2', type='0.5')

penalty.0.5.pivot.acc.3 <- penalty.0.5.acc.3 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '3', type='0.5')

penalty.0.5.acc <-  rbind(
  penalty.0.5.pivot.acc.1,
  penalty.0.5.pivot.acc.2,
  penalty.0.5.pivot.acc.3
)

penalty.0.5 <- penalty.0.5 %>% left_join(
  penalty.0.5.acc
)




# ---------- pivot and join penalty 1e-2 data reversed ----------
penalty.001.r.pivot.loss.1 <- penalty.001.r.loss.1 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '1', type='1e-2_rev')

penalty.001.r.pivot.loss.2 <- penalty.001.r.loss.2 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '2', type='1e-2_rev')

penalty.001.r.pivot.loss.3 <- penalty.001.r.loss.3 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'val_loss') %>% 
  mutate(run = '3', type='1e-2_rev')

penalty.001.r <-  rbind(
  penalty.001.r.pivot.loss.1,
  penalty.001.r.pivot.loss.2,
  penalty.001.r.pivot.loss.3
)

penalty.001.r.pivot.acc.1 <- penalty.001.r.acc.1 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '1', type='1e-2_rev')

penalty.001.r.pivot.acc.2 <- penalty.001.r.acc.2 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '2', type='1e-2_rev')

penalty.001.r.pivot.acc.3 <- penalty.001.r.acc.3 %>%
  pivot_longer(cols = col.names[2:length(col.names)],
               names_to = 'class', 
               values_to = 'acc') %>% 
  mutate(run = '3', type='1e-2_rev')

penalty.001.r.acc <-  rbind(
  penalty.001.r.pivot.acc.1,
  penalty.001.r.pivot.acc.2,
  penalty.001.r.pivot.acc.3
)

penalty.001.r <- penalty.001.r %>% left_join(
  penalty.001.r.acc
)


# --- combine data -------
combined <- rbind(
  baseline,
  penalty.0.5,
  penalty.001,
  penalty.001.r
)

# ------ Worst Performing Class Plot ---------


# use color blind palette

theme_set(theme_minimal())
opts <- options()  # save old options

customPalette <- c("#E31A1C", "#33A02C","#FDBF6F" , "#1F78B4")


worst.peforming.class.loss.plt <- combined %>%
  group_by(type,epoch, run) %>%
  summarize(
    max.loss = max(val_loss)
  ) %>%
  summarize(
    min = min(max.loss),
    med = median(max.loss),
    max = max(max.loss)
  ) %>%
  ggplot(aes(x=epoch, y=med, color=type, fill=type)) +
  #geom_smooth(method="loess", se=FALSE) +
  geom_line(aes(x=epoch, y=med)) +
  geom_ribbon(aes(ymax=max, ymin=min), alpha=0.3, linetype=0) + 
  ylab('Validation Loss') +
  xlab('Epoch') +
  ggtitle('Worst Performing Class Loss') +
  my_theme +
  #theme(plot.title = element_text(hjust=0.5)) +
  guides(fill=guide_legend(title="Penalty"), color=guide_legend(title="Penalty")) +
  scale_fill_manual(values = customPalette) +
  scale_color_manual(values = customPalette)

save_plot('../plots/worst.performing.class.loss.png', plot = worst.peforming.class.loss.plt)

# ------ Overall Loss ------------------
overall.loss.plt <- combined %>% 
  group_by(type, epoch, run, class) %>%
  summarize(
    max.loss = max(val_loss)
  ) %>%
  summarize(
    overall.loss = mean(max.loss)
  ) %>%
  summarize(
    min = min(overall.loss),
    med = median(overall.loss),
    max = max(overall.loss)
  ) %>%
  ggplot(aes(x=epoch, y=med, color=type, fill=type)) +
  geom_line() +
  geom_ribbon(aes(ymax=max, ymin=min), alpha=0.2, linetype=0) +
  ylab('Validation Loss') +
  xlab('Epoch') +
  ggtitle('Overall Validation Loss') +
  #theme(plot.title = element_text(hjust=0.5)) +
  my_theme +
  guides(fill=guide_legend(title="Penalty"), color=guide_legend(title="Penalty")) +
  scale_fill_manual(values = customPalette) +
  scale_color_manual(values = customPalette)
  
save_plot('../plots/overall.loss.png', plot = overall.loss.plt)


# -------- Final Epoch Performance ----------

final.epoch.performance.plt <- combined %>% 
  filter(epoch == 49) %>%
  group_by(type, run, class) %>%
  summarize(
    max.loss = max(val_loss)
  ) %>%
  group_by(type, class) %>%
  summarize(
    min = min(max.loss),
    med = median(max.loss),
    max = max(max.loss)
  ) %>% 
  ggplot(aes(x=class, y=med, color=type, fill=type, group=type)) +
  geom_bar(stat='identity', position = position_dodge(), width = 0.7) +
  geom_errorbar(aes(ymin=min, ymax=max), color='black', width=0.2, position = position_dodge(0.7)) +
  ylab('Validation Loss') +
  xlab('Class') +
  ggtitle('Final Epoch Performance Per Class') +
  #theme(plot.title = element_text(hjust=0.5)) +
  my_theme +
  theme(axis.text.x = element_text(angle = 90, vjust=0.5, hjust = 1)) +
  guides(fill=guide_legend(title="Penalty"), color=guide_legend(title="Penalty")) +
  scale_fill_manual(values = customPalette) +
  scale_color_manual(values = customPalette)

save_plot('../plots/final.epoch.peformance.png', plot = final.epoch.performance.plt)

# ----- Worst Performing Class Accuracy ------
worst.peforming.class.acc.plt <- combined %>%
  group_by(type,epoch, run) %>%
  summarize(
    min.acc = min(acc)
  ) %>%
  summarize(
    min = min(min.acc),
    med = median(min.acc),
    max = max(min.acc)
  ) %>%
  ggplot(aes(x=epoch, y=med, color=type, fill=type)) +
  #geom_smooth(method="loess", se=FALSE) +
  geom_line(aes(x=epoch, y=med)) +
  geom_ribbon(aes(ymax=max, ymin=min), alpha=0.3, linetype=0) + 
  ylab('Accuracy') +
  xlab('Epoch') +
  ggtitle('Worst Performing Class Accuracy') +
  #theme(plot.title = element_text(hjust=0.5)) +
   my_theme +
  guides(fill=guide_legend(title="Penalty"), color=guide_legend(title="Penalty")) +
  scale_fill_manual(values = customPalette) +
  scale_color_manual(values = customPalette)

save_plot('../plots/worst.performing.class.acc.png', plot = worst.peforming.class.acc.plt)

# ------ Training loss ------------

train.combined <- rbind(
  baseline.train.1 %>% mutate(run = '1', type='baseline'),
  baseline.train.2 %>% mutate(run = '2', type='baseline'),
  baseline.train.3 %>% mutate(run = '3', type='baseline'),
  penalty.001.train.1 %>% mutate(run = '1', type='1e-2'),
  penalty.001.train.2 %>% mutate(run = '2', type='1e-2'),
  penalty.001.train.3 %>% mutate(run = '3', type='1e-2'),
  penalty.0.5.train.1 %>% mutate(run = '1', type='0.5'),
  penalty.0.5.train.2 %>% mutate(run = '2', type='0.5'),
  penalty.0.5.train.3 %>% mutate(run = '3', type='0.5'),
  penalty.001.r.train.1 %>% mutate(run ='1', type='1e-2_rev'),
  penalty.001.r.train.2 %>% mutate(run ='2', type='1e-2_rev'),
  penalty.001.r.train.3 %>% mutate(run ='3', type='1e-2_rev')
)

training.loss.plt <- train.combined %>% 
  select(type, epoch, run, val_loss) %>%
  group_by(type, epoch) %>%
  summarize(
    min = min(val_loss),
    med = median(val_loss),
    max = max(val_loss)
  ) %>%
  ggplot(aes(x=epoch, y=med, color=type, fill=type)) +
  geom_line() +
  #geom_ribbon(aes(ymax=max, ymin=min), alpha=0.2, linetype=0) +
  ylab('Validation Loss') +
  xlab('Epoch') +
  ggtitle('Overall Training Loss') +
  #theme(plot.title = element_text(hjust=0.5)) +
  my_theme +
  guides(fill=guide_legend(title="Penalty"), color=guide_legend(title="Penalty")) +
  scale_fill_manual(values = customPalette) +
  scale_color_manual(values = customPalette)


save_plot('../plots/training.loss.png', plot = training.loss.plt)















