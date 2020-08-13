# WESAD_Sensor_Classification_XGBoost
XGBoost classification and analysis of WESAD sensor data.

Start with WESAD_S16_XGBoost_Base_toc.ipynb

## Introduction
Stress related or induced illnesses account for 75% to 90% of doctor office visits and the annual cost to industry is about $300 billion (The American Institute of Stress, 2020).  Effects from long-term stress can produce an increased risk of physiological and psychological disorders including heart complications, diabetes, ulcers, skin problems, loss of sexual desire, changes in appetite, chronic pain, arthritis, depression, increased use of alcohol and drugs, asthma, and anxiety (The American Institute of Stress, 2020).  Emotional stress has been recognized as equally important to heart disease as other risks such as diabetes, hypertension, and smoking (Doolittle, 2019).  In addition, stress has been shown to lower the immune system response resulting in increased susceptibility to flus, colds, and other infections (Doolittle, 2019).  

## Dataset
This project involved using the WESAD (Wearable Stress and Affect Detection) multimodal data to produce multi-class classification reports, which included extensive exploratory data analysis and some data cleaning (Schmidt, et al., 2018a).  The data source was from the UCI machine learning at https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29 (Schmidt, et al., 2018a).  Data included chest-worn and wrist-worn sensor data and scaled self-report questionnaires (Schmidt, et al., 2018a).   Data from the RespiBAN chest-worn device included sensors ECG (electrocardiogram), EDA (electrodermal activity), EMG (electromyogram), TEMP (skin temperature in Celsius), ACC (three-axis acceleration [x, y, z]), and RESP (respiration) (Schmidt, et al., 2018).  Data from the Empatica E4 wrist-worn device included sensors ACC (accelerometer channels), BVP (blood volume pulse), EDA (electrodermal activity), and TEMP (body temperature) (Schmidt, et al., 2018a).  Questionnaires PANAS, STAI, and SAM included scaled emotional responses (Schmidt, et al., 2018a).  The SSSQ questionnaire involved scaled commitment states.  The sensor data was pickle formatted and about 13GB in size.  Data included 15 subjects each in different files.  Subjects included three female and 12 male graduate students.  Row counts were greater than 3.5 million but varied greatly between subject files. 

## Objectives
The focus was on the chest sensor data with the intent to build, test, and compare models and fine tune the best model or models to maximize model accuracy.  The first objective of this project was to test how well XGboost would classify (predict) stress compared to a baseline condition.  The second objective was to compare stress classification between using all chest features and using features from the chest that were equivalent to sensors found in wrist device. Although the number of subjects is too low to make any reliable analysis, analyses included stress between male and female and stress based on comfort level.  The aim, however, was to present examples of what else might be accomplished using this type of data: For example, environment conditions.  Additional goals included a greater understanding sensor data and learning more about XGBoost. 

## Test Models
Test models run were to determine how well some individual features or combination of features would perform using the XGBoost algorithm.  The final model architecture decision occurred after many trial and error experiments and applied to all subsequent models and subjects. All model test runs were in only the WESAD_S16_XGBoost_Base notebook.  Feature (attribute) choices and model runs for remaining 5 subjects had been chosen based-on the outputs from these test runs.  Models used in subsequent notebooks for subjects S4, S8, S11, S13, and S17 included only models allChest and accEdaTemp.  The WESAD_metrics_scores_analysis notebook included statistics and stress related analyses. 

## Methods 
The starting notebook, WESAD_S16_XGBoost_Base, is somewhat busy.  Some of the model runs are comparisons to the base model.  Additional scoring included AUC scores and balanced accuracy, but only in this notebook and not in subsequent notebooks.  Some of the explorations are not necessary but are nice to check out anyway.  It is important for this type of data, however, to look at the series plots to grasp what is occurring and to help with discovery.  Wrist sensors also found in the chest device included an accelerometer, electrodermal activity, and skin temperature. The model accEdaTemp selected from the chest device features resolved the problem of determining presumed wrist performance regarding stress.  In this case, the model accEdaTemp evolved from close examination of the plots shown in Figure 1a and 1b. 
## Figure 1a
<img src="https://github.com/wmadisonf/WESAD_Sensor_Classification_XGBoost/blob/master/Images/series_plots_example.png" width="420" height="360" title="series plot">

## Figure 1b
<img src="https://github.com/wmadisonf/WESAD_Sensor_Classification_XGBoost/blob/master/Images/series_stress_vs_baseline_example.png" width="420" height="360" title="series stress vs baseline">

## XGBoost 
XGBoost is an ensemble of gradient boosted decision trees that performs correlation filtering, automatically reduces the feature set, and is unaffected by outliers (Chen, et al., 2018).    XGBoost is probability-based and unaffected by scale factors, therefore requires no preprocessing such as normalization, standardization, or noise removal.  An advantage to using XGBoost when compared to other algorithms that rely on CPU usage is that it can use a GPU for multithreading parallel computing thereby reducing model run times (Chen, et al., 2018).  
An example of an automatic reduction of the feature set is apparent in the first two model runs, allChest and shortChest.  Feature importance from the allChest (all features) model showed an extremely low influence by the electromyogram attribute, Figure 2.  In the second model run, shortChest, the electromyogram attribute was left out.  However, scores for each model were essentially identical, which apparently shows that feature reduction worked.
## Figure 2
![](Images/feature_importance.png) 

## Metrics
The initial intent was to use the macro average because of high class imbalance.  However, after excluding classes (labels) 5, 6, and 7, the high imbalance had been reduced to a point in which, for example, model allChest macro avg = 0.9845 and f1 avg = 0.9869, scores were similar enough to use the f1 average for the model assessment and f1 scores for comparisons between the baseline and stress classes.  Classes 5, 6, and 7 corresponded to short duration timed periods in which the participant answered questionnaires.  Resulting timed classes included 0, 1, 2, 3, and 4: 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, and 0 = the remaining series data.  An additional option to reduce class imbalance, had it been necessary to keep classes 5, 6, and 7, would have been to combine these three classes to create a single class.  

## Model Assessment 
Changes to the default parameters included changing the objective to softmax for a multiclass model, tree method from ‘hist’ to 'gpu_hist' to make use of the GPU, applying gradient-based to 0.1 for subsampling, and gamma from a default of 0 to 3 to make slightly more conservative.  The evaluation metric used to create the plot, Figure 3, was multiclass error: Train error = 0.01243 and Test error = 0.01310.  Model plot showed excellent performance with no overfitting detected resulting in an f1 accuracy (average) score of 0.9869.  
## Figure 3
<img src="https://github.com/wmadisonf/WESAD_Sensor_Classification_XGBoost/blob/master/Images/merror_plot.png" width="512" height="384" title="merror">

## Analyses can be viewed at: 
[Analysis of scores](https://github.com/wmadisonf/WESAD_Sensor_Classification_XGBoost/blob/master/WESAD_metrics_scores_analysis.ipynb) 

## References
Chen, M, et al. (2018). XGBoost-based algorithm interpretation and application on post-fault transient stability status prediction of power system. Retrieved from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8620201

Doolittle, M. J. (2019). Stress and cancer: An overview. Retrieved from https://med.stanford.edu/survivingcancer/cancer-and-stress/stress-and-cancer.html

Ng, E. (2020).XGBoost for multi-class classification. Retrieved from https://towardsdatascience.com/xgboost-for-multi-class-classification-799d96bcd368
Phaneuf, A. (2020). Latest trends in medical monitoring devices and wearable health technology. retrieved from https://www.businessinsider.com/wearable-technology-healthcare-medical-devices#:~:text=wearable%20healthcare%20technology%3F-,Wearable%20technology%20in%20healthcare%20includes%20electronic%20devices%20that%20consumers%20can,in%202018%2C%20according%20to%20Accenture.

Plog, E. (2018). AUC ROC curve scoring function for multi-class classification. Retrieved from https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659

Schmidt, P., et al. (2018a). Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. Retrieved from https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29.

Schmidt, P., et al. (2018b). Introducing WESAD: A multimodal dataset for wearable stress and affect detection. Retrieved from https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf

The American Institute of Stress. (2020). What is stress? Retrieved from https://www.stress.org/daily-life
  XGBoost (2020). XGBoost parameters. Retrieved from https://xgboost.readthedocs.io/en/latest/parameter.html



