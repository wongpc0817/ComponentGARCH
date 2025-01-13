# HNGARCH for VIX futures pricing

In this project, we implemented both Heston-Nandi GARCH and Component Heston-Nandi GARCH process for Modelling VIX futures. 
 

---
## Experiment Results
The data used for the experiment is not provided here due to confidentiality. However, some brief results are listed here:

<table>
  <caption><strong>Comparison of Normality Test Results for HNGARCH and Component GARCH Residuals</strong></caption>
  <thead>
    <tr>
      <th>Test</th>
      <th>HNGARCH</th>
      <th>Component GARCH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Kolmogorov-Smirnov (K-S) Statistic</td>
      <td>0.4881 (p=0.0)</td>
      <td>0.4878 (p=0.0)</td>
    </tr>
    <tr>
      <td>Jarque-Bera (J-B) Statistic</td>
      <td>53.2599 (p=2.72×10<sup>-12</sup>)</td>
      <td>0.0931 (p=0.9545)</td>
    </tr>
    <tr>
      <td>Anderson-Darling (A-D) Statistic</td>
      <td>3.5501</td>
      <td>0.5317</td>
    </tr>
    <tr>
      <td>Critical Values for A-D</td>
      <td colspan="2">0.575 (15%), 0.655 (10%), 0.786 (5%), 0.916 (2.5%), 1.09 (1%)</td>
    </tr>
    <tr>
      <td>Shapiro-Wilk (S-W) Statistic</td>
      <td>0.9931 (p=7.46×10<sup>-9</sup>)</td>
      <td>0.9992 (p=0.5040)</td>
    </tr>
  </tbody>
</table>
The results suggest that LSGARCH produces residuals closer to normality compared to HNGARCH. This could imply that Component GARCH is better suited for modeling scenarios where normality of residuals is desirable or expected.


---
## Repository Contents
In this repository, you will find:
- `models` folder: contains the main `.py` files for various models. The primary models used in this project are `HNGARCH` and `ComponentGARCH`. Other models are included but may require corrections.
- `main.ipynb`: demonstrates performing modeling and conducting statistical tests on modeling performance.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/wongpc0817/ComponentGARCH.git
   cd ComponentGARCH

2. Install dependencies: 
    ```
    pip install -r requirements.txt
    ```

3. Run the `main.ipynb` notebook.

---
## References
1. Luca Vincenzo Ballestra, Enzo D’Innocenzo, and Christian Tezza. *A GARCH model with two volatility components and two driving factors*, 2024.  
2. Peter Christoffersen, Kris Jacobs, Chayawat Ornthanalai, and Yintian Wang. *Option valuation with long-run and short-run volatility components*. Journal of Financial Economics, 90(3):272–297, 2008.  
3. Jin-Chuan Duan. *The GARCH option pricing model*. Mathematical Finance, 5(1):13–32, 1995.  
4. Steven Heston and Saikat Nandi. *A closed-form GARCH option valuation model*. Review of Financial Studies, 13:585–625, February 2000.  
5. Gaoxiu Qiao, Gongyue Jiang, and Jiyu Yang. *VIX term structure forecasting: New evidence based on the realized semi-variances*. International Review of Financial Analysis, 82:102199, 2022.  
6. Tianyi Wang, Yiwen Shen, Yueting Jiang, and Zhuo Huang. *Pricing the CBOE VIX futures with the Heston–Nandi GARCH model*. Journal of Futures Markets, 37(7):641–659, 2017.  
7. Song-Ping Zhu and Guang-Hua Lian. *An analytical formula for VIX futures and its applications*. Journal of Futures Markets, 32(2):166–190, 2012.  
