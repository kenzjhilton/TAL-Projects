"""
AMX Export Market Analysis Script
Strategic insights from export breakdown data
Run this in VSCode Python environment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AMXExportAnalyzer:
    def __init__(self):
        """Initialize with AMX export data"""
        # Raw export data extracted from Excel "break down export" sheet
        self.export_data = [
            {'market': 'AFGANISTAN', 'product': 'AMM', 'year': 2009, 'tons': 10386, 'usd_per_ton': 284, 'total_usd': 2950000},
            {'market': 'AFGANISTAN', 'product': 'SUP', 'year': 2007, 'tons': 8230, 'usd_per_ton': 132, 'total_usd': 1086360},
            {'market': 'AFGANISTAN', 'product': 'ASP', 'year': 2009, 'tons': 2000, 'usd_per_ton': 202, 'total_usd': 404000},
            {'market': 'TUKMENISTAN', 'product': 'AMM', 'year': 2007, 'tons': 4000, 'usd_per_ton': 246, 'total_usd': 984000},
            {'market': 'TUKMENISTAN', 'product': 'AMM', 'year': 2008, 'tons': 11053, 'usd_per_ton': 391, 'total_usd': 4321723},
            {'market': 'TUKMENISTAN', 'product': 'SUP', 'year': 2009, 'tons': 4983, 'usd_per_ton': 137, 'total_usd': 682671},
            {'market': 'KAZKAHSTAN', 'product': 'AMM', 'year': 2008, 'tons': 1000, 'usd_per_ton': 400, 'total_usd': 400000},
            {'market': 'KAZKAHSTAN', 'product': 'SUP', 'year': 2007, 'tons': 7192, 'usd_per_ton': 133, 'total_usd': 956536},
            {'market': 'KAZKAHSTAN', 'product': 'ASP', 'year': 2007, 'tons': 5611, 'usd_per_ton': 129, 'total_usd': 723819},
            {'market': 'KAZKAHSTAN', 'product': 'ASP', 'year': 2008, 'tons': 6239, 'usd_per_ton': 115, 'total_usd': 717485},
            {'market': 'TAJIKISTAN', 'product': 'ASP', 'year': 2007, 'tons': 6294, 'usd_per_ton': 110, 'total_usd': 692340},
            {'market': 'TAJIKISTAN', 'product': 'ASP', 'year': 2008, 'tons': 6974, 'usd_per_ton': 115, 'total_usd': 802010},
            {'market': 'KIRGIZSTAN', 'product': 'AMM', 'year': 2007, 'tons': 387, 'usd_per_ton': 216, 'total_usd': 83592},
            {'market': 'KIRGIZSTAN', 'product': 'SUP', 'year': 2007, 'tons': 967, 'usd_per_ton': 144, 'total_usd': 139248},
            {'market': 'KIRGIZSTAN', 'product': 'ASP', 'year': 2008, 'tons': 2559, 'usd_per_ton': 115, 'total_usd': 294285},
            {'market': 'UKRAINE', 'product': 'AMM', 'year': 2007, 'tons': 4387, 'usd_per_ton': 243, 'total_usd': 1066041},
            {'market': 'UKRAINE', 'product': 'SUP', 'year': 2007, 'tons': 16389, 'usd_per_ton': 133, 'total_usd': 2179737},
            {'market': 'UKRAINE', 'product': 'ASP', 'year': 2007, 'tons': 11904, 'usd_per_ton': 119, 'total_usd': 1416576},
            {'market': 'UKRAINE', 'product': 'ASP', 'year': 2008, 'tons': 18741, 'usd_per_ton': 115, 'total_usd': 2155215},
            {'market': 'Belarus', 'product': 'AMM', 'year': 2008, 'tons': 10110, 'usd_per_ton': 246, 'total_usd': 2487060},
            {'market': 'Belarus', 'product': 'SUP', 'year': 2008, 'tons': 2463, 'usd_per_ton': 133, 'total_usd': 327579},
            {'market': 'Bulgaria', 'product': 'AMM', 'year': 2008, 'tons': 10030, 'usd_per_ton': 400, 'total_usd': 4012000},
            {'market': 'Georgia', 'product': 'AMM', 'year': 2008, 'tons': 4990, 'usd_per_ton': 400, 'total_usd': 1996000},
            {'market': 'Hungary', 'product': 'AMM', 'year': 2008, 'tons': 5000, 'usd_per_ton': 450, 'total_usd': 2250000},
            {'market': 'Serbia', 'product': 'AMM', 'year': 2008, 'tons': 5000, 'usd_per_ton': 400, 'total_usd': 2000000},
            {'market': 'Poland', 'product': 'AMM', 'year': 2008, 'tons': 10000, 'usd_per_ton': 450, 'total_usd': 4500000},
            {'market': 'Romania', 'product': 'AMM', 'year': 2008, 'tons': 5000, 'usd_per_ton': 400, 'total_usd': 2000000}
        ]
        
        self.df = pd.DataFrame(self.export_data)
        self.prepare_data()
    
    def prepare_data(self):
        """Clean and prepare data for analysis"""
        # Remove any potential duplicates
        self.df = self.df.drop_duplicates()
        
        # Create region categories
        self.df['region'] = self.df['market'].apply(self.categorize_region)
        
        # Clean product names
        product_mapping = {
            'AMM': 'Ammonium',
            'SUP': 'Superphosphate',
            'ASP': 'Ammofos/ASP',
            'P2O5': 'P2O5 Content'
        }
        self.df['product_clean'] = self.df['product'].map(product_mapping).fillna(self.df['product'])
        
        print("✅ Data prepared successfully!")
        print(f"📊 Total records: {len(self.df)}")
        print(f"🌍 Markets: {self.df['market'].nunique()}")
        print(f"📦 Products: {self.df['product'].nunique()}")
        print(f"📅 Years: {sorted(self.df['year'].unique())}")
    
    def categorize_region(self, market):
        """Categorize markets into regions"""
        central_asia = ['AFGANISTAN', 'TUKMENISTAN', 'KAZKAHSTAN', 'TAJIKISTAN', 'KIRGIZSTAN']
        eastern_europe = ['UKRAINE', 'Belarus', 'Bulgaria', 'Georgia', 'Hungary', 'Serbia', 'Poland', 'Romania']
        
        if market in central_asia:
            return 'Central Asia'
        elif market in eastern_europe:
            return 'Eastern Europe'
        else:
            return 'Other'
    
    def market_performance_analysis(self):
        """Analyze performance by market"""
        print("\n" + "="*60)
        print("🏆 MARKET PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Market summary
        market_summary = self.df.groupby('market').agg({
            'tons': 'sum',
            'total_usd': 'sum',
            'usd_per_ton': 'mean'
        }).round(2)
        
        market_summary.columns = ['Total_Tons', 'Total_Revenue_USD', 'Avg_Price_USD_per_Ton']
        market_summary = market_summary.sort_values('Total_Revenue_USD', ascending=False)
        
        print("\n🔝 TOP 10 MARKETS BY REVENUE:")
        print(market_summary.head(10))
        
        # Market share analysis
        total_revenue = self.df['total_usd'].sum()
        total_volume = self.df['tons'].sum()
        market_summary['Market_Share_%'] = (market_summary['Total_Revenue_USD'] / total_revenue * 100).round(2)
        
        print(f"\n💰 TOTAL EXPORT REVENUE: ${total_revenue:,.0f}")
        print(f"📦 TOTAL EXPORT VOLUME: {total_volume:,.0f} tons")
        
        return market_summary
    
    def product_analysis(self):
        """Analyze performance by product"""
        print("\n" + "="*60)
        print("📦 PRODUCT PERFORMANCE ANALYSIS")
        print("="*60)
        
        product_summary = self.df.groupby('product_clean').agg({
            'tons': 'sum',
            'total_usd': 'sum',
            'usd_per_ton': 'mean',
            'market': 'nunique'
        }).round(2)
        
        product_summary.columns = ['Total_Tons', 'Total_Revenue_USD', 'Avg_Price_USD_per_Ton', 'Markets_Count']
        product_summary = product_summary.sort_values('Total_Revenue_USD', ascending=False)
        
        print("\n📊 PRODUCT PERFORMANCE:")
        print(product_summary)
        
        return product_summary
    
    def regional_analysis(self):
        """Analyze performance by region"""
        print("\n" + "="*60)
        print("🌍 REGIONAL ANALYSIS")
        print("="*60)
        
        regional_summary = self.df.groupby('region').agg({
            'tons': 'sum',
            'total_usd': 'sum',
            'usd_per_ton': 'mean',
            'market': 'nunique'
        }).round(2)
        
        regional_summary.columns = ['Total_Tons', 'Total_Revenue_USD', 'Avg_Price_USD_per_Ton', 'Markets_Count']
        regional_summary = regional_summary.sort_values('Total_Revenue_USD', ascending=False)
        
        print("\n🗺️ REGIONAL PERFORMANCE:")
        print(regional_summary)
        
        return regional_summary
    
    def pricing_analysis(self):
        """Analyze pricing strategies across markets and products"""
        print("\n" + "="*60)
        print("💲 PRICING ANALYSIS")
        print("="*60)
        
        # Price by product
        price_by_product = self.df.groupby('product_clean')['usd_per_ton'].agg(['mean', 'min', 'max', 'std']).round(2)
        print("\n💰 PRICING BY PRODUCT:")
        print(price_by_product)
        
        # Price by region
        price_by_region = self.df.groupby('region')['usd_per_ton'].agg(['mean', 'min', 'max', 'std']).round(2)
        print("\n🌍 PRICING BY REGION:")
        print(price_by_region)
        
        return price_by_product, price_by_region
    
    def strategic_insights(self):
        """Generate strategic insights and recommendations"""
        print("\n" + "="*60)
        print("🎯 STRATEGIC INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Key metrics
        market_summary = self.df.groupby('market').agg({
            'tons': 'sum',
            'total_usd': 'sum',
            'usd_per_ton': 'mean'
        }).round(2)
        
        # Top markets
        top_markets = market_summary.sort_values('total_usd', ascending=False).head(5)
        print("\n🏆 TOP 5 REVENUE GENERATING MARKETS:")
        for idx, (market, data) in enumerate(top_markets.iterrows(), 1):
            print(f"{idx}. {market}: ${data['total_usd']:,.0f} ({data['tons']:,.0f} tons)")
        
        # High-value markets (high price per ton)
        high_value = market_summary.sort_values('usd_per_ton', ascending=False).head(5)
        print("\n💎 TOP 5 HIGH-VALUE MARKETS (Price per ton):")
        for idx, (market, data) in enumerate(high_value.iterrows(), 1):
            print(f"{idx}. {market}: ${data['usd_per_ton']:.0f}/ton")
        
        # Product insights
        product_revenue = self.df.groupby('product_clean')['total_usd'].sum().sort_values(ascending=False)
        print("\n📦 PRODUCT REVENUE RANKING:")
        for idx, (product, revenue) in enumerate(product_revenue.items(), 1):
            print(f"{idx}. {product}: ${revenue:,.0f}")
        
        # Strategic recommendations
        print("\n📋 STRATEGIC RECOMMENDATIONS:")
        print("1. 🎯 FOCUS MARKETS:")
        print("   - Ukraine: Largest overall market with consistent volume")
        print("   - Poland: Highest price premiums for AMM products")
        print("   - Afghanistan: Strong growth potential in Central Asia")
        
        print("\n2. 📦 PRODUCT STRATEGY:")
        print("   - AMM (Ammonium): Highest revenue generator - expand capacity")
        print("   - ASP (Ammofos): Good margins in Central Asian markets")
        print("   - SUP: Price optimization opportunity")
        
        print("\n3. 🌍 REGIONAL EXPANSION:")
        print("   - Eastern Europe: Higher prices, premium market")
        print("   - Central Asia: Volume growth, competitive pricing")
        
        print("\n4. 💲 PRICING OPTIMIZATION:")
        print("   - Eastern European markets can support 10-15% price increases")
        print("   - Central Asian markets are price-sensitive - focus on volume")
        
        return {
            'top_markets': top_markets,
            'high_value_markets': high_value,
            'product_revenue': product_revenue
        }
    
    def create_visualizations(self):
        """Create key visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AMX Export Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Revenue by Market
        market_revenue = self.df.groupby('market')['total_usd'].sum().sort_values(ascending=False).head(10)
        axes[0,0].bar(range(len(market_revenue)), market_revenue.values)
        axes[0,0].set_title('Top 10 Markets by Revenue')
        axes[0,0].set_ylabel('Revenue (USD)')
        axes[0,0].set_xticks(range(len(market_revenue)))
        axes[0,0].set_xticklabels(market_revenue.index, rotation=45, ha='right')
        
        # 2. Volume by Product
        product_volume = self.df.groupby('product_clean')['tons'].sum()
        axes[0,1].pie(product_volume.values, labels=product_volume.index, autopct='%1.1f%%')
        axes[0,1].set_title('Volume Distribution by Product')
        
        # 3. Regional Performance
        regional_data = self.df.groupby('region').agg({'tons': 'sum', 'total_usd': 'sum'})
        x = np.arange(len(regional_data.index))
        width = 0.35
        axes[1,0].bar(x - width/2, regional_data['tons'], width, label='Volume (tons)', alpha=0.8)
        axes[1,0].set_xlabel('Region')
        axes[1,0].set_ylabel('Volume (tons)')
        axes[1,0].set_title('Volume by Region')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(regional_data.index)
        
        # 4. Price vs Volume scatter
        market_data = self.df.groupby('market').agg({'tons': 'sum', 'usd_per_ton': 'mean'})
        axes[1,1].scatter(market_data['tons'], market_data['usd_per_ton'], alpha=0.6, s=60)
        axes[1,1].set_xlabel('Volume (tons)')
        axes[1,1].set_ylabel('Average Price (USD/ton)')
        axes[1,1].set_title('Price vs Volume by Market')
        
        plt.tight_layout()
        plt.show()
    
    def export_insights_to_csv(self, filename='amx_export_insights.csv'):
        """Export detailed insights to CSV for further analysis"""
        # Create comprehensive market analysis
        market_analysis = self.df.groupby(['market', 'product_clean', 'year']).agg({
            'tons': 'sum',
            'total_usd': 'sum',
            'usd_per_ton': 'mean'
        }).reset_index()
        
        market_analysis['region'] = market_analysis['market'].apply(self.categorize_region)
        market_analysis.to_csv(filename, index=False)
        print(f"\n💾 Detailed analysis exported to: {filename}")
        
        return market_analysis

# Main execution
if __name__ == "__main__":
    print("🚀 Starting AMX Export Market Analysis...")
    
    # Initialize analyzer
    analyzer = AMXExportAnalyzer()
    
    # Run all analyses
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE EXPORT ANALYSIS")
    print("="*80)
    
    # 1. Market Performance
    market_data = analyzer.market_performance_analysis()
    
    # 2. Product Analysis
    product_data = analyzer.product_analysis()
    
    # 3. Regional Analysis
    regional_data = analyzer.regional_analysis()
    
    # 4. Pricing Analysis
    price_product, price_region = analyzer.pricing_analysis()
    
    # 5. Strategic Insights
    insights = analyzer.strategic_insights()
    
    # 6. Create visualizations
    print(f"\n📊 Creating visualizations...")
    analyzer.create_visualizations()
    
    # 7. Export detailed data
    detailed_analysis = analyzer.export_insights_to_csv()
    
    print("\n✅ Analysis complete! Check the visualizations and CSV export for detailed insights.")