# Step 1: Install required packages (run once) -------------
pkgs_needed = c(
  "readxl",    # to read Excel
  "jsonlite",  # to parse JSON
  "yaml",      # to parse YAML
  "dplyr",     # data wrangling
  "ggplot2",   # plotting
  "purrr",     # map_dbl()
  "janitor",   # clean_names()
  "tidyr",     # pivoting if needed
  "writexl"    # write Excel
)

install.packages(pkgs_needed)

# Step 2: Load libraries -----------------------------------
library(readxl) 
library(jsonlite)
library(yaml)
library(dplyr)
library(ggplot2)
library(purrr)
library(janitor)
library(tidyr)
library(writexl)




setwd(choose.dir())
getwd()
list.files()
# Step 3: Read all inputs ----------------------------------

# 3.1 Market prices
prices = read_excel("(1)PG Market Rates.xlsx", sheet = "PG_Values") %>%
  clean_names()

head(prices)

# 3.2 PG plants (for distances)
pg_plants = read_excel("(2)PG Plants.xlsx", sheet = "Plant_ID") %>%
  clean_names()
head(pg_plants)

# 3.3 Stockpile characteristics
pg_chars = read_excel("(3)PG Stacks.xlsx") %>%
  clean_names()

head(pg_chars)

# 3.4 Capex/opex placeholders
capex = fromJSON("(4)Capex limits.json")
head(capex)

# 3.5 Regulatory limits & tariff
limits = yaml.load_file("(5)Regulation_limits.yaml")
head(limits)

message("Tariff (USD/t·km): ", limits$tariff_usd_per_t_km)

# 3.6 Origin–Destination matrix
od = read_excel("(6)od_matrix_generated.xlsx") %>%
  clean_names() %>%
  mutate(
    distance_km = as.numeric(distance_km)  # ensure numeric
  )

message("OD rows: ", nrow(od), "; preview:")
head(od)

# Step 4: PG‐only margins (smoke‐test) ----------------------

# 4.1 Single PG price
pg_price = prices %>%
  filter(product == "PG") %>%
  pull(price_usd) %>%
  first()

# 4.2 Granulation opex
variable_cost_pg = capex$granulation_line$opex_usd_per_t

# 4.3 Compute per‐pile & total margins
result_pg = pg_chars %>%
  mutate(
    variable_cost_usd_t = variable_cost_pg,
    margin_usd_t        = pg_price - variable_cost_usd_t,
    total_margin_usd    = margin_usd_t * weight_tonnes
  )

message("PG‐only margins (first 5 piles):")
print(
  result_pg %>% 
    select(stockpile_id, weight_tonnes, margin_usd_t, total_margin_usd) %>% 
    head(5)
)

# Plot total margin per stockpile
ggplot(result_pg, aes(stockpile_id, total_margin_usd)) +
  geom_col(fill = "#4C78A8") +
  labs(
    title = "Total PG‐only Margin per Stockpile",
    x     = "Stockpile ID",
    y     = "Total Margin (USD)"
  ) +
  theme_minimal()

# Step 5: Map product codes → JSON keys --------------------
product_to_json_key = list(
  "PG"  = "granulation_line",
  "DAP" = "dap",
  "MAP" = "map",
  "NPK" = "npk"
)

# Step 6: Helper to look up opex per product ---------------
vcost = function(prod_code) {
  key = product_to_json_key[[prod_code]]
  if (is.null(key)) {
    return(NA_real_)
  }
  if (key == "granulation_line") {
    return(capex$granulation_line$opex_usd_per_t)
  }
  # For DAP / MAP / NPK etc., JSON has "variable_cost_usd_per_t"
  purrr::pluck(capex, key, "variable_cost_usd_per_t", .default = NA_real_)
}

# Quick test of vcost()
message("---- Test vcost(): ----")
for (p in names(product_to_json_key)) {
  message(p, " → ", vcost(p))
}

# Step 7: Scenario simulation function ---------------------
simulate_scenario = function(products, lane_row, volume_t = 1) {
  # ensure numeric distance & tariff
  d_km = as.numeric(lane_row$distance_km[1])
  t_km = limits$tariff_usd_per_t_km
  
  haul_cost = if (is.na(d_km)) 0 else d_km * t_km
  lane_label = if (is.na(d_km)) "No Haul" else paste0(lane_row$destination[1], " (", d_km, " km)")
  
  prices %>%
    filter(product %in% products) %>%
    mutate(
      variable_cost_usd_t = purrr::map_dbl(product, vcost),
      haul_cost_usd_t     = haul_cost,
      margin_usd_t        = price_usd - variable_cost_usd_t - haul_cost_usd_t,
      total_margin_usd    = margin_usd_t * volume_t,
      lane                = lane_label
    )
}

# Step 8: Run two demo lanes -------------------------------
demo_products = c("PG", "DAP", "MAP", "NPK")

# match exactly what's in od$destination
lane_A = od %>% filter(destination == "Plant_1")
lane_B = od %>% filter(destination == "Plant_2")

if (nrow(lane_A) == 0) stop("‘Plant_1’ not found in od$destination")
if (nrow(lane_B) == 0) stop("‘Plant_2’ not found in od$destination")

scen_A = simulate_scenario(demo_products, lane_A, volume_t = 1)
scen_B = simulate_scenario(demo_products, lane_B, volume_t = 1)

message("Scenario A (Plant_1):")
print(scen_A)
message("Scenario B (Plant_2):")
print(scen_B)

# Step 9: Visualize multi‐product margins -----------------
all_scen = bind_rows(scen_A, scen_B)

ggplot(all_scen, aes(product, margin_usd_t, fill = lane)) +
  geom_col(position = "dodge") +
  labs(
    title = "Margin per Tonne by Product & Haul Distance",
    x     = "Product",
    y     = "Margin (USD/t)",
    fill  = "Lane"
  ) +
  theme_minimal()
