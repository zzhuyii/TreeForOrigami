# Compute origami design rules using decision tree and random forest

This repo shows the code for computing the origami design rules using a decision tree and random forest. The following two figures explais how the method work. Basically, we turn the inverse design of origami into a binary classification problem and trian a random forest to classify the data that meet the performance target from those that do not. After training the random forest, we pick the most representative tree branches from the enitre random forest for inverse design using the F-score.  

![alt text](https://github.com/zzhuyii/TreeForOrigami/blob/main/Methodology.png)

**Figure 1.** Using decision tree and random forest for inverse design of origami

The binary classification problem is fitting a curve to separate the data that meets the target from those that do not. Most existing ML method will target to fit the boundary between the two groups of data. However, because we are targeting an inverse design problem, it is not necessary to fit the boundary, and instead, we can focus on the core of the data that meets the target performance. Using the F-score to find the most representative tree branches does this job effectively. 

![alt text](https://github.com/zzhuyii/TreeForOrigami/blob/main/Why%20This%20Works.png)

**Figure 2.** Why the proposed method works

This repo is associated with the following reference:

## Reference

Yi Zhu and Evgueni T. Filipov, 2022, Harnessing interpretable machine learning for holistic inverse design of origami, Scientific Reports, 19277.
