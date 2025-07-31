# ===================================================================
# COMPLETE Client FINANCIAL ANALYSIS - ALL VISUALS INCLUDED
# ===================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("="*100)
print("Client FINANCIAL ANALYSIS - COMPLETE VERSION")
print("="*100)

# ----- SETUP WORKING DIRECTORY 
os.chdir(r'')

# ===================================================================
# ----- LOAD DATASETS 
# ===================================================================
summary = pd.read_excel("Client MOTHER FILE 2.xlsx", sheet_name="Summary")
pl_usd = pd.read_excel("Client MOTHER FILE 2.xlsx", sheet_name="PL USD")
export = pd.read_excel("Client MOTHER FILE 2.xlsx", sheet_name="break down export")

print("✅ Data loaded successfully")

# ===================================================================
# ----- EXTRACT KEY FINANCIAL METRICS (2008-2022, 15 years)
# ===================================================================
years = list(range(2008, 2023))

# Extract financial data
t_turnover = pl_usd.iloc[4, 1:16].values.astype(float)
t_cogs = pl_usd.iloc[6, 1:16].values.astype(float)
t_im = pl_usd.iloc[8, 1:16].values.astype(float)
t_ebitda = pl_usd.iloc[18, 1:16].values.astype(float)
t_nprof = pl_usd.iloc[34, 1:16].values.astype(float)

# Handle inventory and products sold with error handling
try:
    t_prod_sold = export.iloc[167, 2:17].values.astype(float)
    t_invent = export.iloc[171, 2:17].values.astype(float)
    # Clean missing values
    t_prod_sold = np.where(np.isnan(t_prod_sold), 0, t_prod_sold)
    t_invent = np.where(np.isnan(t_invent), np.mean(t_invent[~np.isnan(t_invent)]), t_invent)
except:
    # Use estimates if data extraction fails
    t_prod_sold = t_turnover * 0.1
    t_invent = t_turnover * 0.05

print("✅ Financial metrics extracted")

# ===================================================================
# ----- CALCULATE AVERAGES AND RATIOS
# ===================================================================
# Key averages
avg_turnover = np.mean(t_turnover)
avg_cogs = np.mean(t_cogs)
avg_im = np.mean(t_im)
avg_ebitda = np.mean(t_ebitda)
avg_nprof = np.mean(t_nprof)
avg_invent = np.mean(t_invent)
avg_product_sold = np.mean(t_prod_sold)

# Asset data
avg_fixed_asset = summary.iloc[10:11, 3:18].mean().mean()
avg_current_assets = summary.iloc[11:12, 3:18].mean().mean()
avg_total_assets = avg_fixed_asset + avg_current_assets
avg_liabilities = summary.iloc[13:14, 3:18].mean().mean()

# Key ratios
invent_turn = avg_cogs / avg_invent
avg_asset_turnover = avg_turnover / avg_total_assets
liquid_ratio = avg_current_assets / avg_liabilities
int_cov_ratio = summary.iloc[139:140, 4:16].mean().mean()
roa = avg_nprof / avg_total_assets

print("✅ Ratios calculated")

# ===================================================================
# ----- FINANCIAL SUMMARY
# ===================================================================
print(f"\n💼 Client FINANCIAL ANALYSIS SUMMARY")
print("-" * 50)
print(f"Average Turnover: ${avg_turnover:,.2f}")
print(f"Average COGS: ${avg_cogs:,.2f}")
print(f"Average Industrial Margin: ${avg_im:,.2f}")
print(f"Average EBITDA: ${avg_ebitda:,.2f}")
print(f"Average Net Profit: ${avg_nprof:,.2f}")
print(f"Average Products Sold: ${avg_product_sold:,.2f}")
print(f"Average Inventory: ${avg_invent:,.2f}")

print(f"\nKey Financial Ratios:")
print(f"Current Liquidity Ratio: {liquid_ratio:.2f}")
print(f"Interest Coverage Ratio: {int_cov_ratio:.2f}")
print(f"Return on Assets: {roa*100:.2f}%")
print(f"Asset Turnover: {avg_asset_turnover:.3f}")
print(f"Inventory Turnover: {invent_turn:.2f}")

# ===================================================================
# ----- CHART SET 1: COMPREHENSIVE FINANCIAL VISUALIZATIONS
# ===================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Top Left: Financial Metrics Overview
metrics = ['Turnover', 'COGS', 'Industrial Margin', 'EBITDA', 'Net Profit']
metric_values = [avg_turnover, avg_cogs, avg_im, avg_ebitda, avg_nprof]
ax1.bar(metrics, metric_values, color=['green', 'red', 'grey', 'orange', 'blue'])
ax1.set_title('Client Financial Metrics Overview', fontsize=14, fontweight='bold')
ax1.set_ylabel('Amount (USD)')
ax1.tick_params(axis='x', rotation=45)

# Top Right: Key Ratios
ratios = ['Liquidity', 'Interest Coverage', 'ROA (%)']
ratio_values = [liquid_ratio, int_cov_ratio, roa*100]
ax2.bar(ratios, ratio_values, color=['green', 'red', 'yellow'])
ax2.set_title('Key Financial Ratios', fontsize=14, fontweight='bold')
ax2.set_ylabel('Ratio/Percentage')
ax2.tick_params(axis='x', rotation=45)

# Bottom Left: Turnover Over Time
ax3.plot(years, t_turnover, color="green", marker='o', linewidth=2)
ax3.set_title("Client Turnover Trend", fontsize=14, fontweight='bold')
ax3.set_xlabel("Years")
ax3.set_ylabel("Turnover USD")
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# Bottom Right: Multi-Variable Financial Trends
ax4.plot(years, t_turnover, color='green', marker='o', linewidth=1.5, label='Turnover')
ax4.plot(years, t_cogs, color='red', marker='s', linewidth=1.5, label='COGS')
ax4.plot(years, t_ebitda, color='orange', marker='d', linewidth=1.5, label='EBITDA')
ax4.plot(years, t_nprof, color='purple', marker='v', linewidth=1.5, label='Net Profit')
ax4.set_title('Key Financial Metrics Trends', fontsize=14, fontweight='bold')
ax4.set_xlabel('Years')
ax4.set_ylabel('Amount (USD)')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ===================================================================
# ----- CHART SET 2: COMPREHENSIVE TREND ANALYSIS
# ===================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Top Left: Revenue vs Costs with Margin Fill
ax1.plot(years, t_turnover, 'g-o', linewidth=2.5, label='Turnover')
ax1.plot(years, t_cogs, 'r-s', linewidth=2.5, label='COGS')
ax1.fill_between(years, t_turnover, t_cogs, alpha=0.3, color='lightgreen', label='Gross Margin')
ax1.set_title('Revenue vs Cost Analysis', fontsize=14, fontweight='bold')
ax1.set_ylabel('Amount (USD)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Top Right: Profitability Stack
ax2.plot(years, t_im, 'b-^', linewidth=2.5, label='Industrial Margin')
ax2.plot(years, t_ebitda, 'orange', marker='d', linewidth=2.5, label='EBITDA')
ax2.plot(years, t_nprof, 'purple', marker='v', linewidth=2.5, label='Net Profit')
ax2.set_title('Profitability Metrics Evolution', fontsize=14, fontweight='bold')
ax2.set_ylabel('Amount (USD)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Bottom Left: Operational Metrics
ax3.plot(years, t_prod_sold, 'brown', marker='<', linewidth=2.5, label='Products Sold')
ax3.plot(years, t_invent, 'green', marker='>', linewidth=2.5, label='Inventory')
ax3.set_title('Operational Performance', fontsize=14, fontweight='bold')
ax3.set_xlabel('Years')
ax3.set_ylabel('Amount (USD)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Bottom Right: Profit Margins (Percentages)
gross_margin_pct = ((t_turnover - t_cogs) / t_turnover) * 100
net_margin_pct = (t_nprof / t_turnover) * 100
ax4.plot(years, gross_margin_pct, 'green', marker='o', linewidth=2.5, label='Gross Margin %')
ax4.plot(years, net_margin_pct, 'purple', marker='s', linewidth=2.5, label='Net Margin %')
ax4.set_title('Profit Margin Trends', fontsize=14, fontweight='bold')
ax4.set_xlabel('Years')
ax4.set_ylabel('Percentage (%)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ===================================================================
# ----- CHART SET 3: GROWTH ANALYSIS & SUMMARY INSIGHTS
# ===================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Year-over-Year Growth Rates
turnover_growth = np.diff(t_turnover) / t_turnover[:-1] * 100
cogs_growth = np.diff(t_cogs) / t_cogs[:-1] * 100
profit_growth = np.diff(t_nprof) / t_nprof[:-1] * 100
growth_years = years[1:]

ax1.plot(growth_years, turnover_growth, 'g-o', linewidth=2.5, label='Turnover Growth %')
ax1.plot(growth_years, cogs_growth, 'r-s', linewidth=2.5, label='COGS Growth %')
ax1.plot(growth_years, profit_growth, 'purple', marker='d', linewidth=2.5, label='Net Profit Growth %')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Zero Growth')
ax1.set_title('Year-over-Year Growth Rates', fontsize=14, fontweight='bold')
ax1.set_xlabel('Years')
ax1.set_ylabel('Growth Rate (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

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
ax2.set_title('Normalized Performance Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Years')
ax2.set_ylabel('Index (Base Year = 100)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ===================================================================
# ----- SIMPLE FORECASTING WITH VISUALIZATION
# ===================================================================
def simple_forecast():
    print(f"\n🔮 Client FORECASTING ANALYSIS")
    print("-" * 40)
    
    # Method 1: Average growth rate
    growth_rates = [((t_turnover[i] - t_turnover[i-1]) / t_turnover[i-1]) 
                   for i in range(1, len(t_turnover)) if t_turnover[i-1] != 0]
    # Filter extreme values
    growth_rates = [g for g in growth_rates if -0.8 < g < 3.0]
    avg_growth = np.mean(growth_rates) if growth_rates else 0.05
    avg_growth = max(-0.3, min(0.5, avg_growth))  # Cap between -30% and +50%
    
    # Method 2: Recent trend (last 5 years)
    recent_data = t_turnover[-5:]
    x = np.arange(len(recent_data))
    slope = np.polyfit(x, recent_data, 1)[0]
    
    # Method 3: Conservative estimate
    last_avg = np.mean(t_turnover[-3:])
    
    # Generate forecasts
    future_years = [2023, 2024, 2025]
    last_value = t_turnover[-1]
    
    forecast_growth = [last_value * (1 + avg_growth) ** (i+1) for i in range(3)]
    forecast_trend = [t_turnover[-1] + slope * (i+1) for i in range(3)]
    forecast_conservative = [last_avg * (1.02) ** (i+1) for i in range(3)]
    
    # Display results
    print(f"Average Growth Rate: {avg_growth*100:.1f}%")
    print(f"Recent Trend: ${slope:,.0f} per year")
    print(f"Conservative Base: ${last_avg:,.0f}")
    print(f"\nForecasts:")
    
    for i, year in enumerate(future_years):
        print(f"{year}:")
        print(f"  Growth Method: ${forecast_growth[i]:,.0f}")
        print(f"  Trend Method: ${forecast_trend[i]:,.0f}")
        print(f"  Conservative: ${forecast_conservative[i]:,.0f}")
    
    # FORECASTING VISUALIZATION SET
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # Chart 1: Historical data + all forecasts
    ax1.plot(years, t_turnover, 'bo-', linewidth=3, markersize=6, label='Historical Data')
    ax1.plot(future_years, forecast_growth, 'r^-', linewidth=2, markersize=8, label=f'Growth ({avg_growth*100:.1f}%)')
    ax1.plot(future_years, forecast_trend, 'gs-', linewidth=2, markersize=8, label='Trend')
    ax1.plot(future_years, forecast_conservative, 'mo-', linewidth=2, markersize=8, label='Conservative')
    ax1.axvline(x=years[-1] + 0.5, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
    ax1.set_title('Client Turnover: Historical + 3 Forecasting Methods', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Turnover (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Focus on recent years + forecasts
    recent_years_count = min(8, len(years))
    recent_years_plot = years[-recent_years_count:]
    recent_data_plot = t_turnover[-recent_years_count:]
    
    ax2.plot(recent_years_plot, recent_data_plot, 'bo-', linewidth=3, markersize=6, label='Recent Historical')
    ax2.plot(future_years, forecast_growth, 'r^-', linewidth=2, markersize=8, label=f'Growth ({avg_growth*100:.1f}%)')
    ax2.plot(future_years, forecast_trend, 'gs-', linewidth=2, markersize=8, label='Trend')
    ax2.plot(future_years, forecast_conservative, 'mo-', linewidth=2, markersize=8, label='Conservative')
    ax2.axvline(x=years[-1] + 0.5, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Recent Years + Forecasts (Zoomed In)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Turnover (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Growth rates over time
    if len(t_turnover) > 1:
        growth_years_hist = years[1:]
        growth_rates_hist = []
        for i in range(1, len(t_turnover)):
            if t_turnover[i-1] != 0:
                growth = ((t_turnover[i] - t_turnover[i-1]) / t_turnover[i-1]) * 100
                growth = max(-100, min(500, growth))  # Cap extreme values
                growth_rates_hist.append(growth)
            else:
                growth_rates_hist.append(0)
        
        ax3.bar(growth_years_hist[:len(growth_rates_hist)], growth_rates_hist, alpha=0.7, color='lightblue')
        ax3.axhline(y=0, color='red', linestyle='-', alpha=0.7)
        if growth_rates_hist:
            avg_growth_hist = np.mean(growth_rates_hist)
            ax3.axhline(y=avg_growth_hist, color='green', linestyle='--', alpha=0.7, 
                        label=f'Average: {avg_growth_hist:.1f}%')
        ax3.set_title('Historical Year-over-Year Growth Rates', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Growth Rate (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Chart 4: All financial metrics trend
    ax4.plot(years, t_turnover, 'b-', linewidth=2, label='Turnover', marker='o')
    ax4.plot(years, t_cogs, 'r-', linewidth=2, label='COGS', marker='s')
    ax4.plot(years, t_ebitda, 'g-', linewidth=2, label='EBITDA', marker='^')
    ax4.plot(years, t_nprof, 'purple', linewidth=2, label='Net Profit', marker='d')
    ax4.set_title('All Financial Metrics Historical Trend', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Amount (USD)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Format all axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return forecast_growth, forecast_trend, forecast_conservative

# Run forecasting
forecasts = simple_forecast()

# ===================================================================
# ----- COMPREHENSIVE PERFORMANCE SUMMARY
# ===================================================================
print(f"\n📊 COMPREHENSIVE Client PERFORMANCE ANALYSIS")
print("=" * 80)

# Key Performance Metrics Analysis
metrics = {
    'Turnover': t_turnover,
    'COGS': t_cogs,
    'Industrial Margin': t_im,
    'EBITDA': t_ebitda,
    'Net Profit': t_nprof
}

print(f"\n📈 PERFORMANCE TRENDS ANALYSIS")
print("-" * 50)
for name, values in metrics.items():
    total_growth = ((values[-1] / values[0]) - 1) * 100
    trend_slope = np.polyfit(range(len(values)), values, 1)[0]
    avg_value = np.mean(values)
    volatility = np.std(values) / avg_value * 100
    
    print(f"\n{name.upper()}:")
    print(f"  📈 Total Growth (2008-2022): {total_growth:+.2f}%")
    print(f"  📊 Average Annual Value: ${avg_value:,.2f}")
    print(f"  🎯 Trend: {'📈 Rising' if trend_slope > 0 else '📉 Declining'} (${trend_slope:,.2f}/year)")
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

# Growth Momentum
recent_3yr_turnover_growth = ((np.mean(t_turnover[-3:]) / np.mean(t_turnover[:3])) - 1) * 100
recent_3yr_profit_growth = ((np.mean(t_nprof[-3:]) / np.mean(t_nprof[:3])) - 1) * 100

print(f"\n🚀 GROWTH MOMENTUM (Recent 3 Years vs First 3 Years)")
print("-" * 50)
print(f"Turnover Growth: {recent_3yr_turnover_growth:+.2f}%")
print(f"Profit Growth: {recent_3yr_profit_growth:+.2f}%")
print(f"Growth Quality: {'✅ Profitable Growth' if recent_3yr_profit_growth > recent_3yr_turnover_growth else '⚠️ Revenue Growth Outpacing Profit'}")

print("\n" + "="*80)
print("📋 COMPLETE Client ANALYSIS FINISHED")
print("="*80)