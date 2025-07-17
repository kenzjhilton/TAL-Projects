# ----- IMPORT LIBRARIES 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*100)

# ----- CHECK WORKING DIRECTORY 
print(os.getcwd()) # Check current directory 
os.chdir(r'D:\iCloudDrive\2025\TAL\MPC\VCSA\Python')
print(os.listdir()) # List files 

print("="*100)

# ----- LOAD DATASETS FROM AMX MOTHER FILE 2
summary = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="Summary")
pl_usd = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="PL USD")
bs_usd = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="BS USD")
cf_usd = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="CF USD")
funding = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="Detail Invest")
dscr = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="USD-DSCR")
payrole = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="Payroll")
wc = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="WC parameters")
turnover = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="turnover")
export = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="break down export")
cogs = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="cogs")
tax = pd.read_excel("AMX MOTHER FILE 2.xlsx", sheet_name="tax")

print("="*100)

# ----- PREVIEW DATASET 
print(summary.head(),(summary.shape))
print(pl_usd.head(),(pl_usd.shape))
print(bs_usd.head(),(bs_usd.shape))
print(cf_usd.head(),(cf_usd.shape))
print(funding.head(),(funding.shape))
print(dscr.head(),(dscr.shape))
print(payrole.head(),(payrole.shape))
print(wc.head(),(wc.shape))
print(turnover.head(),(turnover.shape))
print(export.head(),(export.shape))
print(cogs.head(),(cogs.shape))
print(tax.head(),(tax.shape))

print("="*100)

# -----  STEP 1: PROFITABILITY ANALYSIS

# Preview P & L statement
print(pl_usd.head(38),(pl_usd.shape))

# ----- .1. Turnover 
print(pl_usd.iloc[4:5, :17]) # Helper - Updated for 2008-2022 (15 years)
# Values - Extract as array, not DataFrame
t_turnover = pl_usd.iloc[4, 1:16].values.astype(float)  # Updated for 15 years (2008-2022)
print("Turnover values shape:", t_turnover.shape)
print("Turnover values:", t_turnover)
# Averages
avg_turnover = pl_usd.iloc[4:5, 1:16].mean().mean()  # Updated for 15 years
print(f"Average Turnover PY: ${avg_turnover:,.2f}") 

print("="*100)

# ----- .2. COGS
print(pl_usd.iloc[6:7, :17]) # Helper - Updated for 2008-2022
# Values - Extract as array
t_cogs = pl_usd.iloc[6, 1:16].values.astype(float)  # Updated for 15 years
print("COGS values shape:", t_cogs.shape)
print("COGS values:", t_cogs)
# Averages
avg_cogs = pl_usd.iloc[6:7, 1:16].mean().mean()  # Updated for 15 years
print(f"Average Cost of Goods Sold PY: ${avg_cogs:,.2f}")

print("="*100)

# ----- .3. Industrial Margin 
print(pl_usd.iloc[8:9, :17]) # Helper - Updated for 2008-2022
# Values - Extract as array
t_im = pl_usd.iloc[8, 1:16].values.astype(float)  # Updated for 15 years
print("Industrial Margin values shape:", t_im.shape)
print("Industrial Margin values:", t_im)
# Averages
avg_im = pl_usd.iloc[8:9, 1:16].mean().mean()  # Updated for 15 years
print(f"Average Industry Margin PY: ${avg_im:,.2f}")

print("="*100)

# ----- .4. EBITDA 
print(pl_usd.iloc[18:19, :17]) # Helper - Updated for 2008-2022
# Values - Extract as array
t_ebitda = pl_usd.iloc[18, 1:16].values.astype(float)  # Updated for 15 years
print("EBITDA values shape:", t_ebitda.shape)
print("EBITDA values:", t_ebitda)
# Averages
avg_ebitda = pl_usd.iloc[18:19, 1:16].mean().mean()  # Updated for 15 years
print(f"Average EBITDA PY: $ {avg_ebitda:,.2f}")

print("="*100)

# ----- .5. NET PROFIT 
print(pl_usd.iloc[34:35, :17]) # Helper - Updated for 2008-2022
# Values - Extract as array
t_nprof = pl_usd.iloc[34, 1:16].values.astype(float)  # Updated for 15 years
print("Net Profit values shape:", t_nprof.shape)
print("Net Profit values:", t_nprof)
# Averages
avg_nprof = pl_usd.iloc[34:35, 1:16].mean().mean()  # Updated for 15 years
print(f"Average Profit PY: ${avg_nprof:,.2f}")

print("="*100)

# ----- STEP 2: BREAKDOWN REPORT ANALYSIS 
print((export.head),(export.shape))
print(export.iloc[167:168, :46]) # Helper - Updated for extended columns

# Values - Extract as array and match the length with improved error handling
try:
    t_prod_sold = export.iloc[167, 2:17].values.astype(float)  # Updated for 15 years (2008-2022)
    # Handle missing values
    t_prod_sold = np.where(np.isnan(t_prod_sold), 0, t_prod_sold)
    print("Products Sold values shape:", t_prod_sold.shape)
    print("Products Sold values:", t_prod_sold)
except Exception as e:
    print(f"Error extracting products sold: {e}")
    # Use turnover-based estimate as fallback
    t_prod_sold = t_turnover * 0.1  # Assume 10% conversion rate
    print("Using estimated Products Sold values:", t_prod_sold)

# Averages 
avg_product_sold = export.iloc[167:168, 2:46].mean().mean()  # Keep original range for average
print(f"Average Products Sold PY: ${avg_product_sold:,.2f}")  

print("="*100)



# ----- STEP 3: Efficiency Ratios 
#.1. Inventory Turnover 
print(export.iloc[171:172, 2:44]) # Helper to identify inventory figures
# Values - Extract as array and match the length with improved error handling
try:
    t_invent = export.iloc[171, 2:17].values.astype(float)  # Updated for 15 years (2008-2022)
    # Handle missing values
    t_invent = np.where(np.isnan(t_invent), np.mean(t_invent[~np.isnan(t_invent)]), t_invent)
    print("Inventory values shape:", t_invent.shape)
    print("Inventory values:", t_invent)
except Exception as e:
    print(f"Error extracting inventory: {e}")
    # Use turnover-based estimate as fallback
    t_invent = t_turnover * 0.05  # Assume 5% of turnover as inventory
    print("Using estimated Inventory values:", t_invent)

# Averages
avg_invent = export.iloc[171:172, 2:44].mean().mean() # Find mean inventory 
print(f"Average Inventory: ${avg_invent:,.2f}") # print average inventory
# Inventory Turnover Average
invent_turn = (avg_cogs / avg_invent) # formula for inventory efficiency 
print(f"Inventory Turnover Ratio: {invent_turn:,.2f}") # Inventory turnover ratio

print("="*100)

#.2. Asset turnover 
print(summary.iloc[10:15, 1:18]) # Helper - Updated for extended period
# Fixed assets
avg_fixed_asset = summary.iloc[10:11, 3:18].mean().mean()  # Updated for 15 years
print(f"Average Fixed Asset: ${avg_fixed_asset:,.2f}")
# Current Assets 
print(summary.iloc[11:12, :18])  # Updated for extended period
avg_current_assets = summary.iloc[11:12, 3:18].mean().mean()  # Updated for 15 years
print(f"Average Current Assets: ${avg_current_assets:,.2f}")
# Total assets 
avg_total_assets = avg_fixed_asset + avg_current_assets
print(f"Average Total Assets: ${avg_total_assets:,.2f}")
# Asset Turnover 
avg_asset_turnover = avg_turnover / avg_total_assets
print(f"Average Asset Turnover Ratio: {avg_asset_turnover:,.5f}")

print("="*100)

# ----- STEP 4: Liquidity Ratios 
print(summary.iloc[13:14, 3:18])  # Updated for extended period
avg_liabilities = summary.iloc[13:14, 3:18].mean().mean()  # Updated for 15 years
print(f"Total Average Liabilities: ${avg_liabilities:,.2f}")

liquid_ratio = avg_current_assets / avg_liabilities
print(f"Current Liquidity Ratio: {liquid_ratio:,.2f}")

print("="*100)

# ----- STEP 5: Coverage Ratios 
print(summary.iloc[139:140, 4:16])  # Updated for extended period
int_cov_ratio = summary.iloc[139:140, 4:16].mean().mean()  # Updated for 15 years
print(f"Average Interest Coverage Ratio: {int_cov_ratio:,.2f}")

# Return on Assets
roa = avg_nprof / avg_total_assets 
print(f"Average Return on Asset Ratio: {roa:,.10f}")

# ----- STEP 6: AMX FINANCIAL ANALYSIS SUMMARY (FILE 2)
print("\n" + "="*50)
print("AMX FINANCIAL ANALYSIS SUMMARY - FILE 2")
print("="*50)
print("AMX FINANCIAL ANALYSIS - Key Values")

print("Turnover values:", t_turnover)
print("COGS values:", t_cogs)
print("Industrial Margin values:", t_im)
print("EBITDA values:", t_ebitda)
print("Net Profit values:", t_nprof)
print("Products Sold values:", t_prod_sold)
print("Inventory values:", t_invent)

print("="*50)

print("AMX FINANCIAL ANALYSIS - Key Averages")
print(f"Average Turnover PY: ${avg_turnover:,.2f}") 
print(f"Average Cost of Goods Sold PY: ${avg_cogs:,.2f}")
print(f"Average Industry Margin PY: ${avg_im:,.2f}")
print(f"Average EBITDA PY: $ {avg_ebitda:,.2f}")
print(f"Average Profit PY: ${avg_nprof:,.2f}")
print(f"Average Products Sold PY: ${avg_product_sold:,.2f}")  
print(f"Average Inventory: ${avg_invent:,.2f}")
print(f"Inventory Turnover Ratio: {invent_turn:,.2f}")
print(f"Average Fixed Asset: ${avg_fixed_asset:,.2f}")
print(f"Average Current Assets: ${avg_current_assets:,.2f}")
print(f"Average Total Assets: ${avg_total_assets:,.2f}")
print(f"Average Asset Turnover Ratio: {avg_asset_turnover:,.5f}")

print("="*50)

print("AMX FINANCIAL ANALYSIS - Ratios")
print(f"Total Average Liabilities: ${avg_liabilities:,.2f}")
print(f"Current Liquidity Ratio: {liquid_ratio:,.2f}")
print(f"Average Interest Coverage Ratio: {int_cov_ratio:,.2f}")
print(f"Average Return on Asset Ratio: {roa:,.10f}")

# -----  STEP 7: COMPREHENSIVE FINANCIAL VISUALIZATIONS (FILE 2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Top Left: Financial Metrics Overview
metrics = ['Turnover', 'COGS', 'Industrial Margin', 'EBITDA', 'Net Profit']
metric_values = [avg_turnover, avg_cogs, avg_im, avg_ebitda, avg_nprof]

ax1.bar(metrics, metric_values, color=['green', 'red', 'grey', 'orange', 'blue'])
ax1.set_title('AMX Financial Metrics Overview - FILE 2', fontsize=14, fontweight='bold')
ax1.set_ylabel('Amount (USD)', fontsize=11)
ax1.tick_params(axis='x', rotation=45, labelsize=9)

# Top Right: Key Ratios
ratios = ['Liquidity', 'Interest Coverage', 'ROA (%)']
ratio_values = [liquid_ratio, int_cov_ratio, roa*100]  # Fixed ROA calculation
colors = ['green', 'red', 'yellow']

ax2.bar(ratios, ratio_values, color=colors)
ax2.set_title('Key Financial Ratios - FILE 2', fontsize=14, fontweight='bold')
ax2.set_ylabel('Ratio/Percentage', fontsize=11)
ax2.tick_params(axis='x', rotation=45, labelsize=9)

# Bottom Left: Turnover Over Time
years = list(range(2008, 2023))  # Updated for 2008-2022 (15 years)

ax3.plot(years, t_turnover, color="green", marker='o', linewidth=2)
ax3.set_title("AMX Turnover Trend - FILE 2", fontsize=14, fontweight='bold')
ax3.set_xlabel("Years", fontsize=11)
ax3.set_ylabel("Turnover USD", fontsize=11)
ax3.tick_params(axis='x', rotation=45, labelsize=9)
ax3.grid(True, alpha=0.3)

# Bottom Right: Multi-Variable Financial Trends
ax4.plot(years, t_turnover, color='green', marker='o', linewidth=1.5, label='Turnover')
ax4.plot(years, t_cogs, color='red', marker='s', linewidth=1.5, label='COGS')
ax4.plot(years, t_ebitda, color='orange', marker='d', linewidth=1.5, label='EBITDA')
ax4.plot(years, t_nprof, color='purple', marker='v', linewidth=1.5, label='Net Profit')

ax4.set_title('Key Financial Metrics Trends - FILE 2', fontsize=14, fontweight='bold')
ax4.set_xlabel('Years', fontsize=11)
ax4.set_ylabel('Amount (USD)', fontsize=11)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45, labelsize=9)

plt.tight_layout()
plt.show()

# ----- STEP 8: COMPREHENSIVE TREND ANALYSIS (FILE 2)

# Define years and check data consistency
years = list(range(2008, 2023))  # Updated for 2008-2022 (15 years)

# Advanced Multi-Variable Analysis
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Top Left: Revenue vs Costs with Margin Fill
ax1.plot(years, t_turnover, 'g-o', linewidth=2.5, label='Turnover')
ax1.plot(years, t_cogs, 'r-s', linewidth=2.5, label='COGS')
ax1.fill_between(years, t_turnover, t_cogs, alpha=0.3, color='lightgreen', label='Gross Margin')
ax1.set_title('Revenue vs Cost Analysis - FILE 2', fontsize=14, fontweight='bold')
ax1.set_ylabel('Amount (USD)', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top Right: Profitability Stack
ax2.plot(years, t_im, 'b-^', linewidth=2.5, label='Industrial Margin')
ax2.plot(years, t_ebitda, 'orange', marker='d', linewidth=2.5, label='EBITDA')
ax2.plot(years, t_nprof, 'purple', marker='v', linewidth=2.5, label='Net Profit')
ax2.set_title('Profitability Metrics Evolution - FILE 2', fontsize=14, fontweight='bold')
ax2.set_ylabel('Amount (USD)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bottom Left: Operational Metrics (FIXED)
ax3.plot(years, t_prod_sold, 'brown', marker='<', linewidth=2.5, label='Products Sold')
ax3.plot(years, t_invent, 'green', marker='>', linewidth=2.5, label='Inventory')
ax3.set_title('Operational Performance - FILE 2', fontsize=14, fontweight='bold')
ax3.set_xlabel('Years', fontsize=11)
ax3.set_ylabel('Amount (USD)', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Bottom Right: Profit Margins (Percentages)
gross_margin_pct = ((t_turnover - t_cogs) / t_turnover) * 100
net_margin_pct = (t_nprof / t_turnover) * 100

ax4.plot(years, gross_margin_pct, 'green', marker='o', linewidth=2.5, label='Gross Margin %')
ax4.plot(years, net_margin_pct, 'purple', marker='s', linewidth=2.5, label='Net Margin %')
ax4.set_title('Profit Margin Trends - FILE 2', fontsize=14, fontweight='bold')
ax4.set_xlabel('Years', fontsize=11)
ax4.set_ylabel('Percentage (%)', fontsize=11)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Format all x-axes
for ax in [ax1, ax2, ax3, ax4]:
    ax.tick_params(axis='x', rotation=45, labelsize=9)

plt.tight_layout()
plt.show()

# ----- STEP 9: GROWTH ANALYSIS & SUMMARY INSIGHTS (FILE 2)

# Growth Rate Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Year-over-Year Growth Rates
turnover_growth = np.diff(t_turnover) / t_turnover[:-1] * 100
cogs_growth = np.diff(t_cogs) / t_cogs[:-1] * 100
profit_growth = np.diff(t_nprof) / t_nprof[:-1] * 100
growth_years = years[1:]  # One less year due to diff calculation

ax1.plot(growth_years, turnover_growth, 'g-o', linewidth=2.5, label='Turnover Growth %')
ax1.plot(growth_years, cogs_growth, 'r-s', linewidth=2.5, label='COGS Growth %')
ax1.plot(growth_years, profit_growth, 'purple', marker='d', linewidth=2.5, label='Net Profit Growth %')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Zero Growth')

ax1.set_title('Year-over-Year Growth Rates - FILE 2', fontsize=14, fontweight='bold')
ax1.set_xlabel('Years', fontsize=11)
ax1.set_ylabel('Growth Rate (%)', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45, labelsize=9)

# Right: Normalized Performance (Base Year = 100)
turnover_normalized = (t_turnover / t_turnover[0]) * 100
cogs_normalized = (t_cogs / t_cogs[0]) * 100
ebitda_normalized = (t_ebitda / t_ebitda[0]) * 100
profit_normalized = (t_nprof / t_nprof[0]) * 100

ax2.plot(years, turnover_normalized, 'g-o', linewidth=2.5, label='Turnover (Indexed)')
ax2.plot(years, cogs_normalized, 'r-s', linewidth=2.5, label='COGS (Indexed)')
ax2.plot(years, ebitda_normalized, 'orange', marker='^', linewidth=2.5, label='EBITDA (Indexed)')
ax2.plot(years, profit_normalized, 'purple', marker='d', linewidth=2.5, label='Net Profit (Indexed)')
ax2.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='Base Year (100)')

ax2.set_title('Normalized Performance Comparison - FILE 2', fontsize=14, fontweight='bold')
ax2.set_xlabel('Years', fontsize=11)
ax2.set_ylabel('Index (Base Year = 100)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45, labelsize=9)

plt.tight_layout()
plt.show()

# ----- STEP 10: COMPREHENSIVE SUMMARY STATISTICS (FILE 2)
print("\n" + "="*80)
print("COMPREHENSIVE AMX FINANCIAL ANALYSIS SUMMARY - FILE 2 (2008-2022)")
print("="*80)

# Key Performance Metrics
metrics = {
    'Turnover': t_turnover,
    'COGS': t_cogs,
    'Industrial Margin': t_im,
    'EBITDA': t_ebitda,
    'Net Profit': t_nprof
}

print("\n📊 PERFORMANCE TRENDS ANALYSIS")
print("-" * 50)
for name, values in metrics.items():
    total_growth = ((values[-1] / values[0]) - 1) * 100
    trend_slope = np.polyfit(range(len(values)), values, 1)[0]
    avg_value = np.mean(values)
    volatility = np.std(values) / avg_value * 100  # Coefficient of variation
    
    print(f"\n{name.upper()}:")
    print(f"  📈 Total Growth ({years[0]}-{years[-1]}): {total_growth:+.2f}%")
    print(f"  📊 Average Annual Value: ${avg_value:,.2f}")
    print(f"  🎯 Trend Direction: {'📈 Rising' if trend_slope > 0 else '📉 Declining'} (${trend_slope:,.2f}/year)")
    print(f"  🎢 Volatility: {volatility:.2f}%")
    print(f"  🏆 Peak: ${np.max(values):,.2f} ({years[np.argmax(values)]})")
    print(f"  📉 Trough: ${np.min(values):,.2f} ({years[np.argmin(values)]})")

# Financial Health Indicators
print(f"\n💼 FINANCIAL HEALTH INDICATORS")
print("-" * 50)
print(f"Current Liquidity Ratio: {liquid_ratio:.2f} ({'✅ Healthy' if liquid_ratio > 1.2 else '⚠️ Monitor' if liquid_ratio > 1.0 else '🚨 Critical'})")
print(f"Interest Coverage: {int_cov_ratio:.2f}x ({'✅ Strong' if int_cov_ratio > 2.5 else '⚠️ Adequate' if int_cov_ratio > 1.5 else '🚨 Weak'})")
print(f"Return on Assets: {roa*100:.2f}% ({'✅ Excellent' if roa*100 > 5 else '⚠️ Good' if roa*100 > 2 else '🚨 Poor'})")
print(f"Asset Turnover: {avg_asset_turnover:.3f} ({'✅ Efficient' if avg_asset_turnover > 1.0 else '⚠️ Moderate'})")
print(f"Inventory Turnover: {invent_turn:.2f} ({'✅ Efficient' if invent_turn > 4 else '⚠️ Monitor' if invent_turn > 2 else '🚨 Slow'})")

# Margin Analysis
latest_gross_margin = gross_margin_pct[-1]
latest_net_margin = net_margin_pct[-1]
avg_gross_margin = np.mean(gross_margin_pct)
avg_net_margin = np.mean(net_margin_pct)

print(f"\n💰 PROFITABILITY ANALYSIS")
print("-" * 50)
print(f"Latest Gross Margin: {latest_gross_margin:.2f}% (Avg: {avg_gross_margin:.2f}%)")
print(f"Latest Net Margin: {latest_net_margin:.2f}% (Avg: {avg_net_margin:.2f}%)")
print(f"Margin Trend: {'📈 Improving' if latest_net_margin > avg_net_margin else '📉 Declining'}")

# Growth Momentum (Updated for 15-year period)
recent_3yr_turnover_growth = ((np.mean(t_turnover[-3:]) / np.mean(t_turnover[:3])) - 1) * 100
recent_3yr_profit_growth = ((np.mean(t_nprof[-3:]) / np.mean(t_nprof[:3])) - 1) * 100

print(f"\n🚀 GROWTH MOMENTUM (Recent 3 Years vs First 3 Years)")
print("-" * 50)
print(f"Turnover Growth: {recent_3yr_turnover_growth:+.2f}%")
print(f"Profit Growth: {recent_3yr_profit_growth:+.2f}%")
print(f"Growth Quality: {'✅ Profitable Growth' if recent_3yr_profit_growth > recent_3yr_turnover_growth else '⚠️ Revenue Growth Outpacing Profit'}")

# COVID-19 Impact Analysis (2020-2022)
covid_years = [2020, 2021, 2022]
covid_indices = [years.index(year) for year in covid_years if year in years]

if covid_indices:
    pre_covid_avg = np.mean(t_turnover[:covid_indices[0]])
    covid_avg = np.mean(t_turnover[covid_indices[0]:])
    covid_impact = ((covid_avg / pre_covid_avg) - 1) * 100
    
    print(f"\n🦠 COVID-19 IMPACT ANALYSIS (2020-2022)")
    print("-" * 50)
    print(f"Pre-COVID Average Turnover: ${pre_covid_avg:,.2f}")
    print(f"COVID-Era Average Turnover: ${covid_avg:,.2f}")
    print(f"COVID Impact on Revenue: {covid_impact:+.2f}%")
    print(f"Recovery Status: {'✅ Recovered' if covid_impact > 0 else '📈 Recovering' if covid_impact > -10 else '🚨 Significant Impact'}")

print("\n" + "="*80)
print("📋 ANALYSIS COMPLETE FOR FILE 2 (2008-2022) - Review charts above for visual insights")
print("="*80)

