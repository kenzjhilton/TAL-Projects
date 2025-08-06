import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class ClientSupplyChain:
    """
    Supply Chain Model for Client (NPK Fertilizer) Production
    Based on the financial model data from Client MOTHER FILE 2.xlsx
    """
    
    def __init__(self):
        # Raw materials and their consumption per ton of Client
        self.bom = {
            'A/Liquid ammonia': 0.321,  # tons per ton of Client
            'Б\\Phosphoric concentrate 28%': 1.24,
            'В\\Sulphuric acid, total': 4.254,
            'Electricity': 0.7,  # MWh per ton
            'Natural gas': 0.389,  # units per ton
            'Compressed air': 1.268,
            'Reused water': 0.129,
            'Industrial water': 0.03
        }
        
        # Raw material costs in UZS per ton (2010 base year)
        self.material_costs = {
            'A/Liquid ammonia': 209999,
            'Б\\Phosphoric concentrate 28%': 306490,
            'В\\Sulphuric acid, total': 41166,
            'sulphur': 45564
        }
        
        # Export markets
        self.markets = [
            'Afghanistan', 'Turkmenistan', 'Kazakhstan', 
            'Tajikistan', 'Kyrgyzstan', 'Ukraine', 'Belarus'
        ]
        
        # Historical production data
        self.historical_data = {
            2008: {'production_tons': 101, 'domestic_sales': 79458.55, 'exports': 54671.29},
            2009: {'production_tons': 74.5, 'domestic_sales': 60309.62, 'exports': 13608},
            2010: {'production_tons': 170.3, 'domestic_sales': 94218, 'exports': 73342.2},
            2011: {'production_tons': 116.3, 'domestic_sales': 90116.8, 'exports': 65978}
        }
        
        # Current inventory levels
        self.inventory = {
            'raw_materials': {},
            'finished_goods': 0,
            'work_in_process': 0
        }
    
    def calculate_material_requirements(self, production_target: float) -> Dict[str, float]:
        """
        Calculate raw material requirements for given production target
        
        Args:
            production_target: Target production in tons of Client
            
        Returns:
            Dictionary of material requirements
        """
        requirements = {}
        for material, consumption_rate in self.bom.items():
            requirements[material] = production_target * consumption_rate
        
        return requirements
    
    def calculate_production_cost(self, production_tons: float) -> Dict[str, float]:
        """
        Calculate total production cost breakdown
        """
        material_req = self.calculate_material_requirements(production_tons)
        costs = {}
        total_material_cost = 0
        
        for material, quantity in material_req.items():
            if material in self.material_costs:
                cost = quantity * self.material_costs[material]
                costs[material] = cost
                total_material_cost += cost
        
        # Add estimated overhead costs (based on model data)
        costs['Labor'] = production_tons * 12230  # UZS per ton
        costs['Utilities'] = production_tons * 4282  # UZS per ton
        costs['Other expenses'] = production_tons * 505  # UZS per ton
        
        costs['Total Material Cost'] = total_material_cost
        costs['Total Production Cost'] = sum(costs.values())
        
        return costs
    
    def optimize_production_plan(self, demand_forecast: Dict[str, float], 
                               capacity_constraint: float = 200) -> Dict:
        """
        Create optimized production plan based on demand forecast
        
        Args:
            demand_forecast: Dict with 'domestic' and 'export' demand in tons
            capacity_constraint: Maximum production capacity in tons
            
        Returns:
            Production plan with costs and material requirements
        """
        total_demand = demand_forecast['domestic'] + demand_forecast['export']
        production_plan = min(total_demand, capacity_constraint)
        
        material_req = self.calculate_material_requirements(production_plan)
        production_costs = self.calculate_production_cost(production_plan)
        
        # Calculate revenue (realistic selling prices in UZS based on historical data)
        # Based on 2010 data: ~1.5M UZS per ton average
        domestic_price = 1200000  # UZS per ton 
        export_price = 1400000    # UZS per ton (higher for exports)
        
        domestic_production = min(demand_forecast['domestic'], production_plan)
        export_production = production_plan - domestic_production
        
        revenue = (domestic_production * domestic_price + 
                  export_production * export_price)
        
        profit = revenue - production_costs['Total Production Cost']
        
        return {
            'production_tons': production_plan,
            'domestic_allocation': domestic_production,
            'export_allocation': export_production,
            'material_requirements': material_req,
            'production_costs': production_costs,
            'revenue': revenue,
            'profit': profit,
            'profit_margin': profit / revenue if revenue > 0 else 0
        }
    
    def inventory_management(self, current_stock: Dict[str, float],lead_times: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate optimal inventory levels and reorder points
        
        Args:
            current_stock: Current inventory levels by material
            lead_times: Lead times in days for each material
            
        Returns:
            Reorder recommendations
        """
        # Calculate safety stock (assume 30 days demand)
        monthly_production = 15  # tons per month (estimated)
        safety_requirements = self.calculate_material_requirements(monthly_production)
        
        reorder_recommendations = {}
        
        for material, safety_qty in safety_requirements.items():
            current_qty = current_stock.get(material, 0)
            lead_time_days = lead_times.get(material, 30)
            
            # Reorder point = (daily usage * lead time) + safety stock
            daily_usage = safety_qty / 30
            reorder_point = (daily_usage * lead_time_days) + safety_qty
            
            if current_qty < reorder_point:
                order_qty = reorder_point * 2 - current_qty  # Order to 2x reorder point
                reorder_recommendations[material] = {
                    'current_stock': current_qty,
                    'reorder_point': reorder_point,
                    'recommended_order': order_qty,
                    'urgency': 'High' if current_qty < safety_qty else 'Medium'
                }
        
        return reorder_recommendations
    
    def export_analysis(self) -> pd.DataFrame:
        """
        Analyze export performance by market
        """
        # Based on historical export data from the model
        export_data = {
            'Market': ['Afghanistan', 'Turkmenistan', 'Kazakhstan', 'Tajikistan'],
            'Historical_Volume_Tons': [2969, 4000, 1500, 800],
            'Price_USD_per_Ton': [115, 120, 118, 112],
            'Market_Share_%': [35, 25, 20, 15]
        }
        
        df = pd.DataFrame(export_data)
        df['Revenue_USD'] = df['Historical_Volume_Tons'] * df['Price_USD_per_Ton']
        
        return df
    
    def supply_chain_dashboard(self):
        """
        Generate supply chain KPIs and metrics
        """
        # Example production scenario
        demand_forecast = {'domestic': 80, 'export': 90}  # tons
        production_plan = self.optimize_production_plan(demand_forecast)
        
        print("=== Client  Client SUPPLY CHAIN DASHBOARD ===\n")
        
        print("PRODUCTION PLAN:")
        print(f"Total Production: {production_plan['production_tons']:.1f} tons")
        print(f"Domestic Allocation: {production_plan['domestic_allocation']:.1f} tons")
        print(f"Export Allocation: {production_plan['export_allocation']:.1f} tons")
        print(f"Revenue: {production_plan['revenue']:,.0f} UZS")
        print(f"Total Cost: {production_plan['production_costs']['Total Production Cost']:,.0f} UZS")
        print(f"Profit: {production_plan['profit']:,.0f} UZS")
        print(f"Profit Margin: {production_plan['profit_margin']:.1%}")
        
        print("\nMATERIAL REQUIREMENTS:")
        for material, qty in production_plan['material_requirements'].items():
            print(f"  {material}: {qty:.2f} units")
        
        print("\nCOST BREAKDOWN (UZS):")
        key_costs = ['Total Material Cost', 'Labor', 'Utilities', 'Total Production Cost']
        for cost_item in key_costs:
            if cost_item in production_plan['production_costs']:
                amount = production_plan['production_costs'][cost_item]
                print(f"  {cost_item}: {amount:,.0f}")
        
        print("\nEXPORT MARKETS ANALYSIS:")
        export_df = self.export_analysis()
        print(export_df.to_string(index=False))
        
        # Inventory recommendations
        current_stock = {
            'A/Liquid ammonia': 50,
            'Б\\Phosphoric concentrate 28%': 200,
            'В\\Sulphuric acid, total': 800
        }
        lead_times = {
            'A/Liquid ammonia': 14,
            'Б\\Phosphoric concentrate 28%': 21,
            'В\\Sulphuric acid, total': 7
        }
        
        reorders = self.inventory_management(current_stock, lead_times)
        if reorders:
            print("\nINVENTORY ALERTS:")
            for material, data in reorders.items():
                print(f"  {material}: {data['urgency']} priority - Order {data['recommended_order']:.1f} units")

# Example usage
if __name__ == "__main__":
    # Initialize the supply chain model
    Clientsupply_chain = ClientSupplyChain()
    
    # Run the dashboard
    Clientsupply_chain.supply_chain_dashboard()
    
    # Example: Calculate requirements for 100 tons production
    print("\n" + "="*50)
    print("EXAMPLE: 100 TONS PRODUCTION REQUIREMENTS")
    requirements = Client_supply_chain.calculate_material_requirements(100)
    for material, qty in requirements.items():
        print(f"{material}: {qty:.2f} units")
    
    # Calculate costs
    costs = Client_supply_chain.calculate_production_cost(100)
    print(f"\nTotal Production Cost: {costs['Total Production Cost']:,.0f} UZS")
    print(f"Cost per Ton: {costs['Total Production Cost']/100:,.0f} UZS")