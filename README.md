# Quantitative Utilities

A growing collection of lightweight, self-contained utilities for **quantitative finance** and **time series machine learning**.  
Each utility is implemented as a standalone, well-documented Python class and demonstrated through accompanying Jupyter notebooks.

---

## 📖 Included Utilities

### 1. TimeSeriesEmbargoCV

A custom cross-validation splitter for time series data with support for an *embargo* period between train and test sets.

- **Embargo support:** Prevents leakage by skipping a configurable embargo window.  
- **Scikit-learn compatible:** Implements `split` and `get_n_splits`.  
- **Use case:** Robust backtesting and leakage-free evaluation for financial ML and other time-dependent problems.
> 📘 Based on an idea introduced by Marcos López de Prado in *Advances in Financial Machine Learning* (2018).

![Time Series Cross-Validation with Embargo](images/EMBARGO.png)

---

### 2. ETFTrick

A utility to simulate a **synthetic ETF** by applying portfolio weights to asset returns (instead of raw price levels).  
This approach avoids the pitfalls of naïve spread calculations and provides a clean, investable-like time series.

- **Flexible inputs:** Works with both `numpy.ndarray` and `pandas.DataFrame`.  
- **Supports shorting:** Negative weights allowed; optional automatic normalization.  
- **Outputs:** Asset returns, cumulative returns, ETF returns, and ETF cumulative returns.  
- **Use case:** Research in portfolio construction, factor testing, or cointegration strategies.
> 📘 Based on an idea introduced by Marcos López de Prado in *Advances in Financial Machine Learning* (2018).


---

## 📂 Project Structure

```
.
├── notebooks/    # Jupyter notebooks demonstrating each utility
├── src/          # Core Python implementations
├── images/       # Visualizations and diagrams used in documentation
├── LICENSE       # MIT License
└── README.md
```

---

## 🔓 Installation

No installation required.  
Simply copy the relevant class from `src/` into your project, or import directly if you clone this repository.

---

## ⚠️ Disclaimer

This project is provided for **educational and research purposes only**.  
It does **not** constitute financial advice or a recommendation to engage in any financial, trading, or investment activities.  
Use at your own risk — the authors assume no liability for direct or indirect damages arising from the use or misuse of this code.

---

## 📜 License

MIT License