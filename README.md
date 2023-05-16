# Sample1
NOTE THAT IT MAY BE NECCESSARY TO PIP INSTALL packages to get the output to properly run. Please read the instructions displayed in your terminal. 

This program reads in the nursery dataset, clusters the dataframe and visualizes using PCA. 
The first portion of the code is focused on exploratory data analysis, and it carefully analyzes the types of data being stored in the csv file. Of particular note is the number of unique values stored in the dataset for each column - this measure explains how applicable a characteristic is for clustering. If a variable is different for all entries, then it isn't useful for the machine learning process - it has no distinguishing features. The same principle applies for variables with only one unique value. 
Once the data has been analyzed and pre-processed, the program calculates the gower's distance for the dataset. 

This gower's distance is used to conduct agglomerative clustering for the dataset, and the different linkage types: single, average, and complete are used to cluster the data. The results are visualized through pca, a dimensionality reduction method used here because of the high dimensionality of the dataset. 

The accuracy of the clustering algorithm is naively computed through silhouette scoring, and this makes it possible to compare the accuracy of each clustering type.
