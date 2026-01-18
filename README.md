# Stock Dashboard Web App

This is a web app that is inspired by popular stock trading apps like **Freetrade**, **Rise by Motilal Oswaal**, and of course, **Yahoo Finance**. In this app, I have created a dynamic dashboard for any company available on Yahoo Finance. 

The app gives user the following information:
- The user can view their chosen stock's performance on a dashboard over different time periods, that they choose
- A **30-day forecast** using the **Holt-Winters Triple Exponential Smoothing** algorithm, and
- A **Recommendations** indicator calculated using Relative Strength Index (**RSI**) and Simple Moving Average (**SMA**) for the respective stocks, that recommends whether a user should Buy, Sell or Hold a stock.

---

I also took this opportunity to practice my **commenting skills** on the `.py` file. So check it out! :) (Although, that file is not updated with recent changes) 

---
You can play around with the app here! - https://stock-dashboard-evlasnoraa.streamlit.app/

Or, you can run it in your own environment using the following steps:

1. Download the `stock_dashboard.py` file  
2. Make sure you are in the same directory as your `.py` file  
3. Use `pip install` command to install all required libraries. Use `pip3 install` if `pip install` doesn't work  
4. Type this into your terminal:

```bash
streamlit run stock_dashboard.py
```
5. Enjoy!
