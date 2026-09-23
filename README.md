<div align="center">

# 🛒 Instacart Market Basket Analysis

**Uncovering purchase patterns, customer segments, and product associations from 3M+ Instacart grocery orders — and predicting what a customer will reorder next.**

![Python](https://img.shields.io/badge/Python-3.7-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

</div>

---

## 📑 Table of Contents

1. [Introduction](#-introduction)
2. [Objectives](#-objectives)
3. [Project Organization](#-project-organization)
4. [Data Description](#-data-description)
5. [Exploratory Data Analysis](#-exploratory-data-analysis)
6. [Customer Segmentation](#-customer-segmentation)
7. [Market Basket Analysis](#-market-basket-analysis)
8. [ML Model to Predict Product Reorders](#-ml-model-to-predict-product-reorders)
9. [Future Work](#-future-work)

---

## 📖 Introduction

Instacart is an American technology company that operates a same-day grocery delivery and pickup service across the U.S. and Canada. Customers shop for groceries through the Instacart mobile app or Instacart.com from various retailer partners, and the order is shopped and delivered by an Instacart personal shopper.

## 🎯 Objectives

- 📊 Analyze the anonymized [Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis/data) of **3 million grocery orders** from more than **200,000 users**
- 🔗 Uncover hidden associations between products for better cross-selling and upselling
- 👥 Perform customer segmentation for targeted marketing and to anticipate customer behavior
- 🤖 Build a machine learning model to predict which previously purchased products will appear in a user's next order

## 📂 Project Organization

```text
.
├── Plots/                                : All generated plots and figures
├── Data Description and Analysis.ipynb   : Initial analysis to understand the data
├── Exploratory Data Analysis.ipynb       : EDA on customer purchase patterns
├── Customers Segmentation.ipynb          : Customer segmentation based on product aisles
├── Market Basket Analysis.ipynb          : Market basket analysis / product associations
├── Feature Extraction.ipynb              : Feature engineering for the ML model
├── Data Preparation.ipynb                : Data preparation for modeling
├── ANN Model.ipynb                       : Neural network model for reorder prediction
├── XGBoost Model.ipynb                   : XGBoost model for reorder prediction
├── LICENSE                               : License
└── README.md                             : Project report
```

---

## 🗃 Data Description

| Table | Highlights |
|---|---|
| **aisles** | 134 unique aisles |
| **departments** | 21 unique departments |
| **products** | 49,688 products, each tagged with an aisle and department |
| **orders** | 3,421,083 orders across 206,209 users, split into Prior / Train / Test sets |
| **order_products__prior** | 3,214,874 orders · 49,677 distinct products · **58.97%** reorder rate |
| **order_products__train** | 131,209 orders · 39,123 distinct products · **59.86%** reorder rate |

**Key findings from `orders`:**
- Total orders per customer range from 0 to 100
- The Train and Test order distributions are similar; the Prior set differs from both
- Days 0 and 1 map to **Saturday** and **Sunday**, and most orders happen during the daytime
- Customers order roughly once a week — peaks at 7, 14, 21, and 30 days since prior order
- **Saturday afternoons** and **Sunday mornings** are the busiest ordering windows

<p align="center">
  <img width="300" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/dow.png" alt="Orders by day of week">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/orders.png" alt="Orders distribution">
  <br>
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/heatmap.png" alt="Day of week vs hour of day heatmap">
</p>

**Key findings from `order_products`:**
- Most orders contain **1–15 items**, with a maximum of 145 items in a single order (true for both Prior and Train sets)

<p align="center">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/prior.png" alt="Items per order, prior set">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/train.png" alt="Items per order, train set">
</p>

---

## 🔍 Exploratory Data Analysis

All individual data files were combined into a single dataframe. Its memory footprint was reduced by **~50% (4.1 GB → 2.0 GB)** through type downcasting, with no information loss.

<details open>
<summary><b>📌 Most popular aisles &amp; departments</b></summary>
<br>

<p align="center">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/popular-aisles.png" alt="Most popular aisles">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/popular-departments.png" alt="Most popular departments">
</p>

Popular departments should be placed close to one another in-store.

</details>

<details open>
<summary><b>📌 Reorder behavior by aisle</b></summary>
<br>

Day-to-day food items have a high reorder percentage, while vitamins, first-aid, and beauty products have a low reorder percentage — groceries are bought regularly, unlike these other categories.

<p align="center">
  <img width="400" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/aisle-high-reorder.png" alt="High reorder aisles">
  <img width="400" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/aisle-low-reorder.png" alt="Low reorder aisles">
</p>

</details>

<details open>
<summary><b>📌 Most popular products &amp; organic trends</b></summary>
<br>

Many of the most popular products are organic. There are fewer organic products overall, but their mean reorder percentage is high — suggesting stores could benefit from stocking more of them.

<p align="center">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/Most-popular-products.png" alt="Most popular products">
</p>
<p align="center">
  <img width="400" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/Total-organic-inorganic-products.png" alt="Organic vs inorganic product counts">
  <img width="400" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/Reorder-organic-inorganic-products.png" alt="Organic vs inorganic reorder rate">
</p>

</details>

<details open>
<summary><b>📌 Cart order, reorders, and loyal buyers</b></summary>
<br>

- The earlier an item is added to the cart, the higher its reorder percentage — consistent with day-to-day essentials being added first
- Reorder percentage vs. number of purchases shows a ceiling effect: many people try a product once and never reorder it, while others buy certain products regularly
- Products with the highest reorder ratio tend to have very few unique buyers (1–15) — a small but loyal group of repeat purchasers

<p align="center">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/Add-to-cart-VS-reorder.png" alt="Add to cart order vs reorder percentage">
</p>
<p align="center">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/reorder-total-orders.png" alt="Reorder percentage vs total orders">
</p>
<p align="center">
  <img width="500" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/reorder-df.png" alt="Reorder ratio vs unique buyers">
</p>

</details>

<details open>
<summary><b>📌 Product coverage</b></summary>
<br>

**85%** of users buy from just **10,000** of the 49,688 total products. For shelf-space optimization, focusing on these ~10,000 products may make sense — assuming the remaining ~39,688 don't contribute significant profit. With pricing data, products could instead be prioritized by revenue, reorder rate, and total sales.

<p align="center">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/cumsum_products.png" alt="Cumulative users per product">
</p>

</details>

---

## 👥 Customer Segmentation

Customers were segmented using the **aisles** they buy from as a proxy for product category, since tracking thousands of individual products per customer isn't practical.

**Method:**
1. **PCA** reduced dimensionality to 10 principal components (KMeans performs poorly on high-dimensional data)
2. **KMeans** clustering was run, with the optimal cluster count (**5**) chosen via the Elbow method

<p align="center">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/elbow.png" alt="Elbow method for optimal k">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/cluster.png" alt="Clusters on first two principal components">
</p>

**Resulting segments:**

| Cluster | Users | Description |
|:---:|---:|---|
| 1 | 5,428 | Very strong preference for water, seltzer, and sparkling water |
| 2 | 55,784 | Mostly order fresh vegetables, followed by fruits |
| 3 | 7,948 | Mostly buy packaged produce and fresh fruits |
| 4 | 37,949 | Very strong preference for fruits, followed by fresh vegetables |
| 5 | 99,100 | Order across many aisles; lower mean orders — likely infrequent or newer users |

---

## 🧺 Market Basket Analysis

Market basket analysis examines which products customers tend to buy together, so retailers can decide what to cross-sell or promote jointly — the term comes from the shopping "basket" a customer fills during a trip. Large retailers like Amazon and Flipkart use it to inform:

- 🏬 Store layout changes based on purchase trends
- 📈 Customer behavior analysis
- 📖 Catalog design
- 🔀 Cross-marketing on online stores
- 📧 Customized emails with add-on sales

Association rule mining is used to surface frequent product co-occurrence patterns using the **Apriori algorithm**, which assumes any subset of a frequent itemset must itself be frequent (e.g., if {Grapes, Apple, Mango} is frequent, then {Grapes, Mango} must be too).

### Metrics

| Metric | Meaning | Formula |
|---|---|---|
| **Support** | Baseline popularity of item A | `Support(A) = transactions with A / total transactions` |
| **Confidence** | Likelihood a customer who bought A also bought B | `Confidence(A⇒B) = Support(A, B) / Support(A)` |
| **Lift** | How much buying A increases the chance of buying B | `Lift(A⇒B) = Confidence(A⇒B) / Support(B)` |

- **Lift = 1** → no correlation between A and B
- **Lift > 1** → A and B are more likely to be bought together
- **Lift < 1** → A and B are unlikely to be bought together

Applying Apriori (via `mlxtend`) to the top 100 most frequent products yielded **28 product pairs (56 rules)** with lift greater than 1.

**Top 10 pairs by lift:**

| Rank | Product A | Product B | Lift |
|:---:|---|---|:---:|
| 1 | Limes | Large Lemons | 3.00 |
| 2 | Organic Strawberries | Organic Raspberries | 2.21 |
| 3 | Organic Avocado | Large Lemon | 2.12 |
| 4 | Organic Strawberries | Organic Blueberries | 2.11 |
| 5 | Organic Hass Avocado | Organic Raspberries | 2.08 |
| 6 | Banana | Organic Fuji Apple | 1.88 |
| 7 | Bag of Organic Bananas | Organic Raspberries | 1.83 |
| 8 | Organic Hass Avocado | Bag of Organic Bananas | 1.81 |
| 9 | Honeycrisp Apple | Banana | 1.77 |
| 10 | Organic Avocado | Organic Baby Spinach | 1.70 |

---

## 🤖 ML Model to Predict Product Reorders

**Goal:** predict which previously purchased products will appear in a user's next order — useful for product recommendations.

### Feature Engineering

<table>
<tr><th width="30%">Group</th><th>Features</th></tr>
<tr>
<td><b>Product-level</b><br><sub>captures a product's popularity</sub></td>
<td>

Average add-to-cart order · total orders · total reorders · reorder % · unique users · organic flag · % of users who bought it a second time

</td>
</tr>
<tr>
<td><b>Aisle / Department-level</b><br><sub>day-to-day vs. once-in-a-while products</sub></td>
<td>

Reorder %, total orders, and reorders per aisle/department · mean & std of add-to-cart order · unique users · binary-encoded aisle & department (avoids one-hot sparsity)

</td>
</tr>
<tr>
<td><b>User-level</b><br><sub>a user's overall purchase pattern</sub></td>
<td>

Mean & std of order day-of-week, hour-of-day, and days-since-prior-order · total orders · total & unique products bought · total reorders · overall reorder % · average order size · mean reordered items per order · reorder % and order count across the user's last 3 orders

</td>
</tr>
<tr>
<td><b>User–Product-level</b><br><sub>how a user treats a specific product</sub></td>
<td>

Avg add-to-cart order & days-since-prior-order for the product · total orders, reorders, and reorder % for the product · order number of the user's last purchase · purchase history for the product across the last 3 orders

</td>
</tr>
</table>

### Modeling

A dataframe was built where each row is a *(user, previously-purchased product)* pair, combining all feature groups above plus current-order context (day-of-week, hour-of-day, etc.). The target `reordered` flags whether the item was ordered again.

- **Memory:** reduced via downcasting; `MinMaxScaler` was used over `StandardScaler`, since the latter required 16 GB of RAM
- **Models:** primarily **XGBoost**, for its ability to handle large data, parallelize, and expose feature importance — plus a **Neural Network** for comparison, since both approaches carry some inherent randomness
- **Class imbalance:** handled via cost-sensitive learning (class weights ≈ `{0: 1, 1: 10}`) rather than SMOTE/upsampling (which grows the dataset) or downsampling (which discards data and introduces bias)
- **Evaluation:** **AUC** was used instead of F1, since F1 can be manipulated by adjusting the classification threshold

Both models performed comparably, with **XGBoost slightly ahead on ROC-AUC**.

<details>
<summary><b>📊 Neural Network — architecture & performance</b></summary>
<br>
<p align="center">
  <img width="400" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/NN%20Architecture.png" alt="NN architecture">
</p>
<p align="center">
  <img width="400" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/NN-Report.png" alt="NN classification report">
</p>
<p align="center">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/NN-Performance.png" alt="NN confusion matrix and ROC">
</p>
</details>

<details open>
<summary><b>📊 XGBoost — performance & feature importance</b></summary>
<br>
<p align="center">
  <img width="400" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/XGBoost-Report.png" alt="XGBoost classification report">
</p>
<p align="center">
  <img width="600" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/XGBoost%20Performance.png" alt="XGBoost confusion matrix and ROC">
</p>
<p align="center">
  <img width="500" src="https://github.com/archd3sai/Instacart-Market-Basket-Analysis/blob/master/Plots/XGBoost%20Feature%20Importance%20Plot.png" alt="XGBoost feature importance">
</p>
</details>

---

## 🚀 Future Work

- Use **collaborative filtering** to recommend products to a customer
