# ✈️ TravelTide: Customer Segmentation & Loyalty Perk Assignment

An end-to-end Data Science project developing a data-driven customer segmentation framework for **TravelTide**. This repository demonstrates a complete workflow from SQL data extraction and exploratory data analysis to heuristic baselines, unsupervised machine learning (K-Means & DBSCAN), and PCA dimensionality reduction.

---

## 📌 Project Overview & Business Goal

Marketing at TravelTide requires assigning **5 distinct, mutually exclusive loyalty perks** (e.g., Free Checked Bags, Free Hotel Meals, Flight Discounts) to customers to boost retention and increase Lifetime Value (LTV). 

To achieve this, we developed a multi-stage data pipeline that contrasts classical rule-based heuristics with machine learning clustering to justify the final algorithmic approach.

### Key Objectives:
1. **Data Extraction & Aggregation:** Aggregate raw session, flight, hotel, and demographic data via SQL into user-level features.
2. **Heuristic Baseline Analysis (`02b`):** Evaluate non-ML techniques (Categorization, Thresholding, Bargain Index) to understand rule-based limits.
3. **ML Pipeline Execution (`03` & `04`):** Scale behavioral features using `MinMaxScaler` and train deterministic **K-Means ($K=5$)** models alongside **DBSCAN** for outlier detection.
4. **Dimensionality Reduction & Validation (`05`):** Apply **PCA** to visualize the 5 high-dimensional clusters in a 2D feature space and justify feature variance.

---

## 📁 Repository Structure

```text
traveltide-customer-segmentation/
├── data/
│   ├── traveltide_features_unscaled.csv   # Aggregated raw user features
│   └── traveltide_features_scaled.csv     # MinMax-normalized features
├── notebooks/
│   ├── 01_data_extraction.ipynb          # SQL aggregation & data loading
│   ├── 02_data_exploration.ipynb         # General exploratory data analysis (EDA)
│   ├── 02b_heuristic_baseline.ipynb      # Non-ML baseline & Bargain Index analysis
│   ├── 03_preprocessing.ipynb            # Scaling (MinMaxScaler) & feature prep
│   ├── 04_kmeans_dbscan_clustering.ipynb # K-Means & DBSCAN pipeline
│   └── 05_pca_dimensionality_reduction.ipynb # PCA 2D visualization & Scree plots
├── reports/
│   ├── figures/                          # Saved diagnostic and visual artifacts
│   └── executive_summary.md              # Final business recommendations
├── README.md                             # Project documentation
└── requirements.txt                      # Python dependencies
