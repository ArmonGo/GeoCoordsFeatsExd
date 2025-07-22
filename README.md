# GeoCoordsFeatsExd
This is the execution of extended paper titled "Coordinates Are Just Features: A Benchmark Study of Geospatial Modeling"

**Abstract**

Geospatial inference has long been recognized as a critical topic of research. Modeling approaches in this area can generally be categorized into two main types, i.e. explicit and implicit spatial dependence learning. The key difference between these categories lies in whether spatial information (typically coordinates) is used as input to a distance function or simply treated as standard features in a machine learning algorithm.
Traditional geospatial statistical models, such as Geographically Weighted Regression (GWR) and Kriging, explicitly model spatial dependence. However, they often suffer from high computational costs and struggle to balance the trade-off between predictive performance and efficiency. In this work, we aim to demonstrate that explicitly modeling geospatial dependence is often not necessary. Treating coordinates as standard input features can yield competitive predictive performance while significantly reducing computational overhead, provided that a sufficiently capable learner is used.
To substantiate our claims, we conduct an extensive comparison across a wide range of models. As an extended version of our previous work, we broaden the scope of models considered and include additional tabular deep learning models based on the transformer architecture and its attention mechanism. We also assess the statistical significance of performance differences across datasets. Furthermore, we include an interpretability analysis to examine the role of coordinates in models that learn spatial information either explicitly or implicitly.
Our results show that even models which treat coordinates as standard features can achieve competitive performance, with substantially lower training costs, while still effectively capturing spatial dependence. To the best of our knowledge, this is the first comprehensive study to evaluate both the effectiveness and efficiency of using coordinate inputs directly in spatial prediction tasks 


**Dataset**

All the datasets used in this paper are currently available in kaggle. Five datasets come from R package Spatstat.data.

More implementation details please check the file 'dataloader.py' and 'dataload.R'.

| Dataset  | Link  |
|----------|-------------------------------------------------------------|
| Beijing  | [Link](https://www.kaggle.com/datasets/ruiqurm/lianjia/)  |
| Dubai    | [Link](https://www.kaggle.com/datasets/azharsaleem/real-estate-goldmine-dubai-uae-rental-market)  |
| London   | [Link](https://www.kaggle.com/datasets/jakewright/house-price-data)  |
| Melbourne | [Link](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)  |
| New York | [Link](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market)  |
| Paris    | [Link](https://www.kaggle.com/datasets/benoitfavier/immobilier-france)  |
| Perth    | [Link](https://www.kaggle.com/datasets/syuzai/perth-house-prices)  |
| Seattle  | [Link](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)  |
| Yield  | [Link](https://geodacenter.github.io/data-and-lab/lasrosas/)  |


