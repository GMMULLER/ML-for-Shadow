# ML-for-Shadow

This repository contains the source code used in the final project required for earning the title: Bachelor of Computer Science at Federal Fluminense University (UFF/RJ).  

This project involves the determination of accumulated shadow data in urban environments using Machine Learning. The project was inspired by [Shadow Accrual Map](https://github.com/VIDA-NYU/shadow-accrual-maps) and used its data as training data. 

The source code is divided into two folders: "data-loading" and "models". The first one, contains scripts to transform and format data from different cities. The second one, contains scripts for training and testing all created models.

## Abstract

With the perspective of growing in the next years, many big cities have been fighting
to maintain citizens’ quality of life through a well planned growth. In that context, one
of the biggest challenges in urban planning is the management of shadow casting from
buildings, since it can affect heat and sunlight distribution throughout the city. This chal-
lenge is being explored by tools (used by specialists) that try to predict the impact of the
construction, modification and demolition of urban structures. These tools, however, are
computationally expensive and usually limited to small contexts and time spans. That
being the case, the present work proposes a Machine Learning model that uses Artificial
Neural Networks (MLP) to predict shadow accumulation more efficiently, using precalcu-
lated shadow data. The model is the result of iterative evolutions of a initial solution that
used Polynomial Regression. The evolutions were built through hyperparameter optimi-
zation, feature engineering and error investigation. In the end of this process the model
was evaluated in New York with the following scores: 58,89 (RMSE in minutes), 0,87
(R2) and 38,62 (MAE in minutes). Yet, the model does not have a suficient temporal
and geographical generalization, which means that it can not be used (with the same
success) in any city of the globe or day of the year. Solving these problems envolves the
utilization of a more complex model, higher quality and quantity of data and the use of
pre and post processing.  

Keywords: Machine Learning, Neural Network, MLP, Shadow, Cities, Urban Planning,
Sunlight, Quality of life, Shadow Accumulation  
