# ===================================================================
# ----- IMPORT LIBRARIES 
# ===================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*100)

# ----- CHECK WORKING DIRECTORY 
print(os.getcwd()) # Check current directory 
os.chdir(r'')
print(os.listdir()) # List files 
print("="*100)

# ===================================================================
# ----- LOAD DATASETS FROM Client MOTHER FILE 2
# ===================================================================
summary = pd.read_excel("Client MOTHER FILE 2.xlsx", sheet_name="Summary")
pl_usd = pd.read_excel("Client MOTHER FILE 2.xlsx", sheet_name="PL USD")
bs_usd = pd.read_excel("Client MOTHER FILE 2.xlsx", sheet_name="BS USD")
cf_usd = pd.read_excel("Client MOTHER FILE 2.xlsx", sheet_name="CF USD")
dscr = pd.read_excel("Client MOTHER FILE 2.xlsx", sheet_name="USD-DSCR")
turnover = pd.read_excel("Client MOTHER FILE 2.xlsx", sheet_name="turnover")


print("="*100)
# ===================================================================
# ----- PREVIEW DATASET 
# ===================================================================
print(summary.head(),(summary.shape))
print(pl_usd.head(),(pl_usd.shape))
print(bs_usd.head(),(bs_usd.shape))
print(cf_usd.head(),(cf_usd.shape))
print(dscr.head(),(dscr.shape))
print(turnover.head(),(turnover.shape)) # In millions UZS 
print("="*100)


# ===================================================================
# ----- Step 1: Set  P&L Variables 
# ===================================================================

print(summary.iloc[4:5, :7]) # Helper 

# Actual Values 
actual_rev = summary.iloc[2:3, 3:7].astype(float)  # Actual revenue
print(actual_rev)

actual_cogs = summary.iloc[3:4, 3:7].astype(float)  # Actual COGS
print(actual_cogs)

actual_op_cost = summary.iloc[4:5, 3:7].astype(float)  # Actual operating costs
print(actual_op_cost)

actual_net_profit = summary.iloc[5:6, 3:7].astype(float)  # Actual net profit
print(actual_net_profit)

# Forecast Values 

forecast_rev = summary.iloc[2:3, 7:15].astype(float)  # Forecast revenue
print(forecast_rev)

forecast_cogs = summary.iloc[3:4, 7:15].astype(float)  # Forecast COGS
print(forecast_cogs)    

forecast_op_cost = summary.iloc[4:5, 7:15].astype(float)  # Forecast operating costs
print(forecast_op_cost)

forecast_net_profit = summary.iloc[5:6, 7:15].astype(float)  # Forecast net profit
print(forecast_net_profit)


# ===================================================================
# ----- Step 2: Set  BS Variables 
# ===================================================================

print(summary.iloc[11:12, :]) # Helper 

# Actual Values
actual_fixed_assets = summary.iloc[10:11, 3:7].astype(float)  # Actual fixed assets
print(actual_fixed_assets)

actual_current_assets = summary.iloc[11:12, 3:7].astype(float)  # Actual current assets
print(actual_current_assets)

actual_equity = summary.iloc[12:13, 3:7].astype(float)  # Actual equity
print(actual_equity)

actual_liabilities = summary.iloc[13:14, 3:7].astype(float)  # Actual liabilities
print(actual_liabilities)

# Forecast Values 
forecast_fixed_assets = summary.iloc[10:11, 7:15].astype(float)  # Forecast fixed assets
print(forecast_fixed_assets)

forecast_current_assets = summary.iloc[11:12, 7:15].astype(float)  # Forecast current assets
print(forecast_current_assets)

forecast_equity = summary.iloc[12:13, 7:15].astype(float)  # Forecast equity
print(forecast_equity)

forecast_liabilities = summary.iloc[13:14, 7:15].astype(float)  # Forecast liabilities
print(forecast_liabilities)

# ===================================================================
# ----- Step 3: Set  CF Variables 
# ===================================================================

print(summary.iloc[26:27, 4:7]) # Helper 

# Actual Values 

actual_total_cf = summary.iloc[26:27, 4:7].astype(float)  # Actual total cash flow
print(actual_total_cf)

forecast_total_cf = summary.iloc[26:27, 7:15].astype(float)  # Forecast total cash flow
print(forecast_total_cf)


# ===================================================================
# ----- Step 4: Variable Dashboards 
# ===================================================================

print("="*100)
print("Key Summary Statistics")
print("="*100)

def print_summary_statistics(): # Function to print summary statistics
    print("Actual Revenue:\n", actual_rev)
    print("Actual COGS:\n", actual_cogs)
    print("Actual Operating Costs:\n", actual_op_cost)
    print("Actual Net Profit:\n", actual_net_profit)
    
    print("\nForecast Revenue:\n", forecast_rev)
    print("Forecast COGS:\n", forecast_cogs)
    print("Forecast Operating Costs:\n", forecast_op_cost)
    print("Forecast Net Profit:\n", forecast_net_profit)

    print("\nActual Fixed Assets:\n", actual_fixed_assets)
    print("Actual Current Assets:\n", actual_current_assets)
    print("Actual Equity:\n", actual_equity)
    print("Actual Liabilities:\n", actual_liabilities)

    print("\nForecast Fixed Assets:\n", forecast_fixed_assets)
    print("Forecast Current Assets:\n", forecast_current_assets)
    print("Forecast Equity:\n", forecast_equity)
    print("Forecast Liabilities:\n", forecast_liabilities)

    print("\nActual Total Cash Flow:\n", actual_total_cf)
    print("Forecast Total Cash Flow:\n", forecast_total_cf)
print_summary_statistics()

# ===================================================================
# ----- Step 5: Visualize Key Metrics
# ===================================================================

def plot_key_metrics():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Key Financial Metrics', fontsize=16 , fontweight='bold')
    sns.barplot(x=actual_rev.columns, y=actual_rev.values.flatten(), ax=ax1, color='blue', label='Actual Revenue')
    sns.barplot(x=forecast_rev.columns, y=forecast_rev.values.flatten(), ax=ax2, color='orange', label='Forecast Revenue')
    sns.barplot(x=actual_cogs.columns, y=actual_cogs.values.flatten(), ax=ax3, color='blue', label='Actual COGS')
    sns.barplot(x=forecast_cogs.columns, y=forecast_cogs.values.flatten(), ax=ax4, color='orange', label='Forecast COGS')
    ax1.set_title('Actual Revenue')
    ax2.set_title('Forecast Revenue')
    ax3.set_title('Actual COGS')   
    ax4.set_title('Forecast COGS')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Months')
        ax.set_ylabel('Amount (in millions UZS)')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()
plot_key_metrics()

