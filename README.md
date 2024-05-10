# Module Data Cleaning and Anomaly detection

This Module assists in data cleaning by imputing missing values in your tabular data and detecting anomalies in your data by using a Large language models (LLMs).
We use the embedding models to vector our data and represent in higher dimensionality, then we cluster and index it using ANNOY algorithm. Our missing values are imputed from the nearest neighbour points in 'n' dimensional space. 

## Data Imputation
In this mode, the module returns a csv file with processed values and additional columns of nearest neighbours.

![image](https://github.com/mogith-pn/Module-Data-Cleanser/assets/143642606/8c72b007-2852-40d8-934e-a8843524ddd1)


## Anomaly detection
The module also effectively finds the anomaly in your data and label it. This mode is activated by default when your uploaded data doesn't have any null values.
We will be using the embedding model to embed the data and use ["DBSCAN"](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#demo-of-dbscan-clustering-algorithm) .
Then we'll be using UMAP algorithm to reduce the dimensionality of the data to represent it in 2D scatter plot for visualization.

![image](https://github.com/mogith-pn/Module-Data-Cleanser/assets/143642606/cbe7ed99-23c3-45fc-b0e5-cb4c6d93856b)

![image](https://github.com/mogith-pn/Module-Data-Cleanser/assets/143642606/3d5e552b-881f-4e42-b2c0-06f9ccefab69)
