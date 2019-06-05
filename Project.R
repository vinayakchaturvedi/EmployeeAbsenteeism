#Clear the environment
rm(list=ls())

#set the working directory
setwd(dir = "C:/Users/vinayak/Desktop/EmployeeAbsenteesm")

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees',"usdm","randomForest","e1071","plyr", "dplyr")

install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#Load the dataset
dataset = readxl::read_excel("Absenteeism_at_work_Project.xls")
dataset = as.data.frame(dataset)
removeCharacter <- function(x) {colnames(x) <- gsub("/", " per ", colnames(x));x}
spaceless <- function(x) {colnames(x) <- gsub(" ", "_", colnames(x));x}
dataset <- removeCharacter(dataset)
dataset <- spaceless(dataset)
################################## Exploratory Data Analysis ################################################

#Check the structure of the dataset
str(dataset)

dataset$Reason_for_absence[dataset$Reason_for_absence %in% 0] = NA

#Univariate Analysis and Variable Consolidation --> Transform into proper data type
factor_col_no = c(1,2,3,4,5,12,13,14,15,16,17)
dataset[,factor_col_no] <- lapply(dataset[,factor_col_no] , factor)

#Check the structure of the dataset
str(dataset)

##################################Missing value analysis################################################
sum(is.na(dataset))

#There are 178 missing values in the dataset so we need to perform missing value analysis.
missing_val = data.frame(apply(dataset,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(dataset)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
write.csv(missing_val, "Mising_perc.csv", row.names = F)

ggplot(data = missing_val[1:3,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
  geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
  ggtitle("Missing data percentage (EmployeeAbsenteeism)") + theme_bw()

#To replace the missing values there are 3 ways
#1. KNN, 2. Mean, 3. Median
#dataset[1, 6] = 289
#dataset[1, 6] = NA
#Mean Method
#dataset$`Transportation expense`[is.na(dataset$`Transportation expense`)] = mean(dataset$`Transportation expense`, na.rm = T)

#Median Method
#dataset$`Transportation expense`[is.na(dataset$`Transportation expense`)] = median(dataset$`Transportation expense`, na.rm = T)

# kNN Imputation
dataset = knnImputation(dataset, k = 3)
sum(is.na(dataset))

#Actual #dataset[1, 6] = 289
#Mean = 220.9426
#Median = 225
#KNN = 289

################################## Analyze Data Insights (Distribution) ##########################################
summary(dataset)

numeric_index = sapply(dataset,is.numeric) #selecting only numeric
numeric_data = dataset[,numeric_index]

factor_index = sapply(dataset,is.factor)  #selecting only factor
factor_data = dataset[,factor_index]

cnames_numeric = colnames(numeric_data)
cnames_factor = colnames(factor_data)

#Remove target variable from cnames_numeric
cnames_numeric <- cnames_numeric[!cnames_numeric %in% "Absenteeism_time_in_hours"]

#Analyze Distribution
for(i in 1:length(cnames_numeric)) {
  assign(paste0("gn",i), ggplot(data = dataset, aes_string(x = cnames_numeric[i])) + geom_histogram(bins = 25, fill="green", col="black")+ ggtitle(paste("Histogram of",cnames_numeric[i])))
}

gridExtra::grid.arrange(gn1,gn2,gn3,gn4,ncol=2)
gridExtra::grid.arrange(gn5,gn7,gn8,gn9,ncol=2)

################################## Outlier Analysis ################################################
for (i in 1:length(cnames_numeric)) {
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames_numeric[i]), x = "Absenteeism_time_in_hours"), data = subset(dataset))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames_numeric[i],x="Absenteeism_time_in_hours")+
           ggtitle(paste("Box plot of AbsenteeismTime for",cnames_numeric[i])))
}
#
## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn7,ncol=3)
gridExtra::grid.arrange(gn8,gn9,ncol=3)

for(i in cnames_numeric) {
  val = dataset[,i][dataset[,i] %in% boxplot.stats(dataset[,i])$out]
  #print(length(val))
  dataset[,i][dataset[,i] %in% val] = NA
}
sum(is.na(dataset))

#Impute NA using KNN impute
dataset = knnImputation(dataset, k = 3)
dataset['Absenteeism_time_in_hours'] <-  round(dataset['Absenteeism_time_in_hours'], 0)

################################## Feature Selection ################################################
## Correlation Plot 
corrgram(dataset[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

#As we can see in the plot weight and body mass index are very highly +ve correlated so we can remove 1 of them 
#And weight is less related to AbsenteeismHour as compared to body mass index so I am removing weight
dataset=dataset[, !(colnames(dataset) %in% c("Weight"))]

#Chi-square test for correlation between categorical variable
for (i in 1:length(cnames_factor)) {
  for(j in i+1:length(cnames_factor)) {
    if(j<=length(cnames_factor)) {
      print(paste(names(factor_data)[i], " VS ", names(factor_data)[j]))
      print(chisq.test(table(factor_data[,i],factor_data[,j])))
    }
  }
}

# P-Value relation between categorical variable (only those who have p-value<0.05)
# "Social_drinker  VS  Social_smoker" : 0.00406
# "Education  VS  Social_smoker": 2.2e-16
# "Education  VS  Social_drinker":2.2e-16
# "Disciplinary_failure  VS  Social_smoker"= 0.003241
# "Month_of_absence  VS  Social_smoker"= 0.02312
# Here we can see that social smoker is dependent on almost all other independent variable
# So we can remove it
dataset=dataset[, !(colnames(dataset) %in% c("Social_smoker"))]

################################## Feature Scaling ################################################
#Normality check
# qqnorm(dataset$Transportation_expense)
# hist(dataset$Transportation_expense)
# 
# numeric_index = sapply(dataset,is.numeric) #selecting only numeric except AbsenteeismTime
# numeric_data = dataset[,numeric_index]
# 
# cnames_numeric = colnames(numeric_data)
# cnames_numeric = cnames_numeric[-11]
# #Apply Normalization
# for(i in cnames_numeric){
#   print(i)
#   dataset[,i] = (dataset[,i] - min(dataset[,i]))/
#     (max(dataset[,i] - min(dataset[,i])))
# }
#No need to apply feature scaling as in our problem we need to identify the reason for absenteeism 
#nothing to predict here (Human Readable -- Actual Values)
################################## Result ################################################
#Clean the environment
rmExcept("dataset")

numeric_index = sapply(dataset,is.numeric) #selecting only numeric
numeric_data = dataset[,numeric_index]

factor_index = sapply(dataset,is.factor)  #selecting only factor
factor_data = dataset[,factor_index]

cnames_numeric = colnames(numeric_data)
cnames_factor = colnames(factor_data)

#selecting only numeric except Absenteeism_time_in_hours
cnames_numeric = cnames_numeric[-9]

#Que1: What changes company should bring to reduce the number of absenteeism?
data  =  group_by(dataset,dataset$Absenteeism_time_in_hours)
result1=summarise(data
               , Transportation_expense          =mean(Transportation_expense)
               , Distance_from_Residence_to_Work = mean(Distance_from_Residence_to_Work)
               , Service_time                    = mean(Service_time)
               , Age                             = mean(Age)
               , Work_load_Average_per_day       = mean(Work_load_Average_per_day)
               , Hit_target                      = mean(Hit_target)
               , Height                          = mean(Height)
               , Body_mass_index                 = mean(Body_mass_index)
               , Count = n())

ggplot(data = dataset, aes(x=Reason_for_absence, y= Absenteeism_time_in_hours)) + geom_bar(stat = 'identity') 

ggplot(data = dataset, aes(x=Month_of_absence, y= Absenteeism_time_in_hours)) + geom_bar(stat = 'identity') 

ggplot(data = dataset, aes(x=Seasons, y= Absenteeism_time_in_hours)) + geom_bar(stat = 'identity') 

ggplot(data = dataset, aes(x=Social_drinker, y= Absenteeism_time_in_hours)) + geom_bar(stat = 'identity') 

ggplot(data = dataset, aes(x=Day_of_the_week, y= Absenteeism_time_in_hours)) + geom_bar(stat = 'identity') 

#Que2:  How much losses every month can we project in 2011 if same trend of absenteeism continues? 
data  =  group_by(dataset,dataset$Month_of_absence)
result2=summarise(data, Absenteeism_time_in_hours=mean(Absenteeism_time_in_hours), Count = n())
