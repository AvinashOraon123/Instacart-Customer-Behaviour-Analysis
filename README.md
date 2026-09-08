# 🛒 Instacart Market Basket Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-Model-orange.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen.svg" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
</p>

<p align="center">
  <em>Uncovering purchase patterns, customer segments, and product associations from 3M+ Instacart grocery orders — and predicting what a customer will reorder next.</em>
</p>

---

## 📑 Table of Contents

- [Introduction](#-introduction)
- [Objectives](#-objectives)
- [Project Organization](#-project-organization)
- [Data Description](#-data-description)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Customer Segmentation](#-customer-segmentation)
- [Market Basket Analysis](#-market-basket-analysis)
- [ML Model to Predict Product Reorders](#-ml-model-to-predict-product-reorders)
- [Future Work](#-future-work)
- [License](#-license)

---

## 📖 Introduction

Instacart is an American technology company that operates a same-day grocery delivery and pickup service across the U.S. and Canada. Customers shop for groceries through the Instacart mobile app or Instacart.com from various retailer partners, and the order is shopped and delivered by an Instacart personal shopper.

## 🎯 Objectives

- Analyze the anonymized [dataset](https://www.kaggle.com/c/instacart-market-basket-analysis/data) of 3 million grocery orders from more than 200,000 Instacart users, open-sourced by Instacart
- Uncover hidden associations between products for better cross-selling and upselling
- Perform customer segmentation for targeted marketing and to anticipate customer behavior
- Build a machine learning model to predict which previously purchased products will be in a user's next order

## 📂 Project Organization

```
.
├── Plots/                                      : Contains all plots
├── Data Description and Analysis.ipynb         : Initial analysis to understand the data
├── Exploratory Data Analysis.ipynb             : EDA to analyze customer purchase patterns
├── Customers Segmentation.ipynb                : Customer segmentation based on product aisles
├── Market Basket Analysis.ipynb                : Market basket analysis to find product associations
├── Feature Extraction.ipynb                    : Feature engineering and extraction for the ML model
├── Data Preparation.ipynb                      : Data preparation for modeling
├── ANN Model.ipynb                             : Neural network model for product reorder prediction
├── XGBoost Model.ipynb                         : XGBoost model for product reorder prediction
├── LICENSE                                     : License
└── README.md                                   : Project report
```

<br />

## 🗃 Data Description

- **aisles** — Contains 134 unique aisles.

- **departments** — Contains 21 unique departments.

- **orders** — All orders made by different users. Key findings:
    - A total of 3,421,083 orders were made by 206,209 users.
    - There are three sets of orders — Prior, Train, and Test. The distributions of orders in the Train and Test sets are similar, whereas the distribution in the Prior set differs.
    - Total orders per customer range from 0 to 100.
    - Based on the "Orders vs. Day of Week" plot, days 0 and 1 map to Saturday and Sunday respectively, assuming most people buy groceries on weekends.
    - The majority of orders are made during the daytime.
    - Customers order roughly once a week, supported by peaks at 7, 14, 21, and 30 in the "Orders vs. Days Since Prior Order" graph.
    - The "Day of Week" × "Hour of Day" heatmap shows Saturday afternoons and Sunday mornings are prime ordering times.

<p align="center">
  <img width="300" height="200" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/dow.png">
</p>

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/orders.png">
</p>

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/heatmap.png">
</p>

- **products** — A total of 49,688 products, each tagged with an aisle and a department. Product counts vary widely across aisles and departments.

- **order_products_prior** — Which products were ordered, the order they were added to the cart, and whether each was a reorder.
    - Contains 3,214,874 orders covering 49,677 distinct products.
    - The "Count vs. Items in Cart" plot shows most people buy 1–15 items per order, with a maximum of 145 items in a single order.
    - The reorder rate in this set is 58.97%.

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/prior.png">
</p>

- **order_products_train** — Same structure as above, for the training set.
    - Contains 131,209 orders covering 39,123 distinct products.
    - Most people buy 1–15 items per order, with a maximum of 145 items in a single order.
    - The reorder rate in this set is 59.86%.

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/train.png">
</p>

## 🔍 Exploratory Data Analysis

For the analysis, all separate data files were combined into a single dataframe. To fit it in memory, its size was reduced by ~50% (4.1 GB → 2.0 GB) via type conversion, without losing any information.

- **Most popular aisles**, based on total products bought:

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/popular-aisles.png">
</p>

- Day-to-day food items have a high reorder percentage, while items like vitamins, first-aid, and beauty products have a low reorder percentage — makes sense, since groceries are bought regularly but those other items are not.

<p align="center">
<img width="400" height="220" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/aisle-high-reorder.png">
<img width="400" height="220" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/aisle-low-reorder.png">
</p>

- **Most popular departments.** Store layouts should place popular departments close to one another.

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/popular-departments.png">
</p>

- **Most popular products** — notably, many are organic.

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/Most-popular-products.png">
</p>

- There are fewer organic products overall, but their mean reorder percentage is high — suggesting stores could benefit from stocking more organic products.

<p align="center">
    <img width="400" height="250" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/Total-organic-inorganic-products.png"/>
    <img width="400" height="250" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/Reorder-organic-inorganic-products.png"/>
</p>

- Plotting add-to-cart order against mean reorder percentage shows that the earlier an item is added to the cart, the higher its reorder percentage — consistent with day-to-day essentials being added first.

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/Add-to-cart-VS-reorder.png">
</p>

- Reorder percentage vs. number of product purchases shows a ceiling effect: many people try a product once and never reorder it, while others buy certain products regularly.

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/reorder-total-orders.png">
</p>

- Products with the highest reorder ratio tend to have very few unique buyers (1–15) — a small but loyal group of repeat purchasers.

<p align="center">
  <img width="500" height="400" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/reorder-df.png">
</p>

- The cumulative-users-per-product plot shows that 85% of users buy from just 10,000 of the 49,688 total products. For shelf-space optimization, focusing on these ~10,000 products may make sense — assuming the remaining ~39,688 products don't contribute significant profit. With pricing data, products could instead be prioritized by revenue, reorder rate, and total sales.

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/cumsum_products.png">
</p>

## 👥 Customer Segmentation

Customer segmentation groups customers by common characteristics so companies can market to each group effectively. Segmentation here is based on the aisles customers buy from — since there are thousands of products and customers, aisles are used as a proxy for product categories.

Principal Component Analysis (PCA) was applied to reduce dimensionality (KMeans performs poorly on high-dimensional data). Using 10 principal components, KMeans clustering was performed with the optimal number of clusters (5) selected via the Elbow method:

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/elbow.png">
</p>

The clusters visualized along the first two principal components:

<p align="center">
  <img width="600" height="400" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/cluster.png">
</p>

The clustering produces 5 distinct segments. After examining the most frequent products in each:

| Cluster | Users | Description |
|---|---|---|
| 1 | 5,428 | Very strong preference for water, seltzer, and sparkling water |
| 2 | 55,784 | Mostly order fresh vegetables, followed by fruits |
| 3 | 7,948 | Mostly buy packaged produce and fresh fruits |
| 4 | 37,949 | Very strong preference for fruits, followed by fresh vegetables |
| 5 | 99,100 | Order across many aisles; lower mean orders — likely infrequent or newer users |

## 🧺 Market Basket Analysis

Market basket analysis is a modeling technique based on the theory that if you buy a certain group of items, you're more or less likely to buy another group of items. It helps retailers understand purchase behavior for cross-selling, up-selling, sales promotions, loyalty programs, store design, and discount planning.

It scrutinizes the products customers tend to buy together and uses that information to decide which products should be cross-sold or promoted together — the term comes from the shopping carts ("baskets") shoppers fill during a trip.

Association rule mining is used to find associations between objects in a set — frequent patterns in a transaction database, relational database, or other information repository.

The most common approach, market basket analysis, is used by large retailers like Amazon and Flipkart to analyze buying habits by finding associations between items placed together in a "shopping basket." These associations can inform:

- Store layout changes based on purchase trends
- Customer behavior analysis
- Catalog design
- Cross-marketing on online stores
- Customized emails with add-on sales, etc.

### Metrics

**Support** — The default popularity of an item. The support of item A is the ratio of transactions involving A to the total number of transactions.

**Confidence** — The likelihood that a customer who bought A also bought B. It's the ratio of transactions involving both A and B to transactions involving B.
- Confidence(A ⇒ B) = Support(A, B) / Support(B)

**Lift** — The increase in the sale of A when B is also sold.
- Lift(A ⇒ B) = Confidence(A, B) / Support(B)

- Lift(A ⇒ B) = 1 → no correlation between A and B.
- Lift(A ⇒ B) > 1 → positive correlation — A and B are more likely to be bought together.
- Lift(A ⇒ B) < 1 → negative correlation — A and B are unlikely to be bought together.

**Apriori Algorithm** — Assumes that any subset of a frequent itemset must itself be frequent; it's the algorithm behind market basket analysis here. For example, if a transaction containing {Grapes, Apple, Mango} is frequent, then {Grapes, Mango} must also be frequent.

The Apriori algorithm from the `mlxtend` Python library was applied to the top 100 most frequent products, yielding 28 product pairs (56 rules total) with lift greater than 1. The top 10 pairs by lift:

| Product A | Product B | Lift |
|---|---|---|
| Limes | Large Lemons | 3.00 |
| Organic Strawberries | Organic Raspberries | 2.21 |
| Organic Avocado | Large Lemon | 2.12 |
| Organic Strawberries | Organic Blueberries | 2.11 |
| Organic Hass Avocado | Organic Raspberries | 2.08 |
| Banana | Organic Fuji Apple | 1.88 |
| Bag of Organic Bananas | Organic Raspberries | 1.83 |
| Organic Hass Avocado | Bag of Organic Bananas | 1.81 |
| Honeycrisp Apple | Banana | 1.77 |
| Organic Avocado | Organic Baby Spinach | 1.70 |

## 🤖 ML Model to Predict Product Reorders

This anonymized transactional data can be used to predict which previously purchased products will appear in a user's next order — useful for product recommendations.

Building the model required extracting features that capture a user's purchase pattern and each product's popularity:

**Product-level features** — capture a product's popularity among users
```
(1)  Product's average add-to-cart order
(2)  Total times the product was ordered
(3)  Total times the product was reordered
(4)  Reorder percentage of a product
(5)  Total unique users of a product
(6)  Whether the product is organic
(7)  Percentage of users who bought the product a second time
```

**Aisle- and department-level features** — capture whether an aisle/department leans toward day-to-day products (vegetables, fruit, soda, water) or once-in-a-while products (medicine, personal care)
```
(8)  Reorder percentage, total orders, and reorders of a product's aisle
(9)  Mean and std of aisle add-to-cart order
(10) Aisle unique users
(11) Reorder percentage, total orders, and reorders of a product's department
(12) Mean and std of department add-to-cart order
(13) Department unique users
(14) Binary encoding of the aisle feature (avoids the sparsity of one-hot encoding)
(15) Binary encoding of the department feature (avoids the sparsity of one-hot encoding)
```

**User-level features** — capture a user's purchase pattern and behavior
```
(16) User's average and std day-of-week of order
(17) User's average and std hour-of-day of order
(18) User's average and std days-since-prior-order
(19) Total orders by a user
(20) Total products a user has bought
(21) Total unique products a user has bought
(22) User's total reordered products
(23) User's overall reorder percentage
(24) Average order size of a user
(25) User's mean reordered items across all orders
(26) Percentage of reordered items in a user's last three orders
(27) Total orders among a user's last three orders
```

**User-product-level features** — capture how a user orders/reorders specific products
```
(28) User's avg add-to-cart order for a product
(29) User's avg days-since-prior-order for a product
(30) User's total orders, reorders, and reorder percentage for a product
(31) Order number when the user last bought the product
(32) User's purchase history for the product across their last three orders
```

### ML Models

Using these extracted features, a dataframe was built containing, for every product a user has previously bought: user-level, product-level, aisle/department-level, and user-product-level features, plus current-order context (day-of-week, hour-of-day, etc.). The target, `reordered`, indicates whether each previously purchased item was ordered again.

Because the dataframe was large, memory usage was reduced via downcasting. `MinMaxScaler` was preferred over `StandardScaler`, since the latter required 16 GB of RAM. Standard modeling practice was followed, relying primarily on **XGBoost** for its ability to handle large data, parallelize, and provide feature importance. A **Neural Network** was also built for comparison, to account for randomness inherent in both approaches. To handle class imbalance, cost-sensitive learning was used with class weights of roughly `{0: 1, 1: 10}`, rather than random upsampling/SMOTE (which would increase data size) or random downsampling (which discards potentially important information and introduces bias).

Since the F1 score can be manipulated by adjusting the classification threshold, **AUC** was used for model evaluation instead. Performance for both models is shown below via confusion matrix, ROC curve, and classification report, along with XGBoost's feature importance plot. Both models perform similarly, with XGBoost slightly ahead on ROC-AUC.

**Neural Network — Architecture & Performance:**

<p align="center">
  <img width="400" height="200" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/NN%20Architecture.png">
</p>

<p align="center">
  <img width="400" height="200" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/NN-Report.png">
</p>

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/NN-Performance.png">
</p>

**XGBoost — Performance & Feature Importance:**

<p align="center">
  <img width="400" height="200" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/XGBoost-Report.png">
</p>

<p align="center">
  <img width="600" height="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/XGBoost%20Performance.png">
</p>

<p align="center">
  <img width="500" height="750" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/XGBoost%20Feature%20Importance%20Plot.png">
</p>

## 🚀 Future Work

- Utilize collaborative filtering to recommend products to a customer.

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
