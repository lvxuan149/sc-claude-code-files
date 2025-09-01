# E-commerce Business Analytics Dashboard

A comprehensive business intelligence solution featuring both Jupyter notebook analysis and a professional Streamlit dashboard for e-commerce sales data with configurable time periods and reusable business metrics calculations.

## Overview

This project transforms a basic exploratory data analysis into a professional, maintainable business intelligence framework. The refactored solution provides:

- **Configurable Analysis**: Easily analyze any time period or compare different years
- **Modular Architecture**: Reusable data loading and metrics calculation modules
- **Professional Visualizations**: Business-oriented charts with proper formatting
- **Strategic Insights**: Automated generation of business recommendations

## Project Structure

```
data_analysis/
├── EDA_Refactored.ipynb     # Main analysis notebook
├── dashboard.py             # Streamlit dashboard application
├── data_loader.py           # Data loading and processing module
├── business_metrics.py      # Business metrics calculation module
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── ecommerce_data/         # Data directory
    ├── orders_dataset.csv
    ├── order_items_dataset.csv
    ├── products_dataset.csv
    ├── customers_dataset.csv
    ├── order_reviews_dataset.csv
    └── order_payments_dataset.csv
```

## Features

### 1. Configurable Analysis Framework
- Set analysis year, comparison year, and month filters
- Flexible time period analysis without code changes
- Automatic handling of missing data periods

### 2. Comprehensive Business Metrics
- **Revenue Analysis**: Total revenue, growth rates, average order value
- **Product Performance**: Category analysis, revenue share, top performers
- **Geographic Insights**: State-level revenue and order analysis
- **Customer Satisfaction**: Review scores, satisfaction distribution
- **Delivery Performance**: Delivery times, speed categorization

### 3. Professional Visualizations
- Monthly revenue trend charts
- Product category performance bars
- Interactive geographic heatmaps
- Customer satisfaction distributions
- Consistent color schemes and formatting

### 4. Automated Insights
- Strategic recommendations based on data patterns
- Performance benchmarking and alerts
- Executive summary generation

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab (for notebook analysis)

### Installation Steps

1. **Clone or download the project files**

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure data files are in place**:
   - Place CSV files in the `ecommerce_data/` directory
   - Verify all required files are present (see Project Structure above)

4. **Run the applications**:

   **For Streamlit Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

   **For Jupyter Notebook Analysis**:
   ```bash
   jupyter notebook EDA_Refactored.ipynb
   ```

## Usage Guide

### Streamlit Dashboard

1. **Launch the dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

2. **Navigate the interface**:
   - Use the **year filter** in the top-right to select analysis period
   - View **KPI cards** showing key metrics with trend indicators
   - Explore **interactive charts** in the 2x2 grid layout
   - Monitor **customer experience metrics** in the bottom row

3. **Dashboard Features**:
   - **Real-time filtering**: All charts update automatically when year is changed
   - **Professional styling**: Clean, business-ready interface
   - **Trend indicators**: Green/red arrows showing performance changes
   - **Formatted values**: Currency displayed as $300K, $2M for readability

### Notebook Analysis

1. **Open the refactored notebook**: `EDA_Refactored.ipynb`

2. **Configure analysis parameters** in the first code cell:
   ```python
   ANALYSIS_YEAR = 2023        # Year to analyze
   COMPARISON_YEAR = 2022      # Comparison year (optional)
   ANALYSIS_MONTH = None       # Specific month or None for full year
   DATA_PATH = 'ecommerce_data/'
   ```

3. **Run all cells** to generate the complete analysis

### Advanced Configuration

#### Analyzing Specific Time Periods
```python
# Analyze only Q4 2023
for month in [10, 11, 12]:
    ANALYSIS_MONTH = month
    # Run analysis
```

#### Custom Data Paths
```python
# Use different data location
DATA_PATH = '/path/to/your/data/'
```

#### Filtering by Order Status
```python
# Modify in data_loader.py create_sales_dataset method
status_filter = 'delivered'  # or 'shipped', 'processing', etc.
```

### Module Usage

#### Data Loading Module
```python
from data_loader import EcommerceDataLoader, load_and_process_data

# Quick start
loader, processed_data = load_and_process_data('ecommerce_data/')

# Advanced usage
loader = EcommerceDataLoader('ecommerce_data/')
loader.load_raw_data()
processed_data = loader.process_all_data()

# Create filtered dataset
sales_data = loader.create_sales_dataset(
    year_filter=2023,
    month_filter=None,
    status_filter='delivered'
)
```

#### Business Metrics Module
```python
from business_metrics import BusinessMetricsCalculator, MetricsVisualizer

# Calculate metrics
metrics_calc = BusinessMetricsCalculator(sales_data)
report = metrics_calc.generate_comprehensive_report(
    current_year=2023,
    previous_year=2022
)

# Create visualizations
visualizer = MetricsVisualizer(report)
revenue_fig = visualizer.plot_revenue_trend()
category_fig = visualizer.plot_category_performance()
```

## Key Business Metrics

### Revenue Metrics
- **Total Revenue**: Sum of all delivered order item prices
- **Revenue Growth Rate**: Year-over-year percentage change
- **Average Order Value (AOV)**: Average total value per order
- **Monthly Growth Trends**: Month-over-month performance

### Product Performance
- **Category Revenue**: Revenue by product category
- **Market Share**: Percentage of total revenue by category
- **Category Diversity**: Distribution across product lines

### Geographic Analysis
- **State Performance**: Revenue and order count by state
- **Market Penetration**: Number of active markets
- **Regional AOV**: Average order value by geographic region

### Customer Experience
- **Review Scores**: Average satisfaction rating (1-5 scale)
- **Satisfaction Distribution**: Percentage of high/low ratings
- **Delivery Performance**: Average delivery time and speed metrics

## Output Examples

### Console Output
```
BUSINESS METRICS SUMMARY - 2023
============================================================

REVENUE PERFORMANCE:
  Total Revenue: $3,360,294.74
  Total Orders: 4,635
  Average Order Value: $724.98
  Revenue Growth: -2.5%

CUSTOMER SATISFACTION:
  Average Review Score: 4.10/5.0
  High Satisfaction (4+): 84.2%

DELIVERY PERFORMANCE:
  Average Delivery Time: 8.0 days
  Fast Delivery (≤3 days): 28.5%
```

### Generated Visualizations
- Monthly revenue trend line charts
- Top product category horizontal bar charts
- Interactive US state choropleth maps
- Customer satisfaction distribution charts

## Customization Options

### Adding New Metrics
1. Extend the `BusinessMetricsCalculator` class in `business_metrics.py`
2. Add visualization methods to `MetricsVisualizer` class
3. Update the notebook to display new metrics

### Custom Visualizations
```python
# Example: Custom visualization
def plot_custom_metric(self, data):
    fig, ax = plt.subplots(figsize=(12, 6))
    # Your visualization code
    return fig
```

### Data Source Modifications
- Modify `data_loader.py` to handle different CSV structures
- Update column mappings in the `EcommerceDataLoader` class
- Add new data validation rules

## Troubleshooting

### Common Issues

1. **Module Import Errors**:
   - Ensure all files are in the same directory
   - Check Python path configuration

2. **Missing Data Files**:
   - Verify CSV files are in the `ecommerce_data/` directory
   - Check file naming matches expected patterns

3. **Empty Results**:
   - Verify date filters match available data
   - Check order status filtering

4. **Visualization Issues**:
   - Ensure all required packages are installed
   - Check Plotly version compatibility for interactive maps

### Performance Optimization
- For large datasets, consider chunked processing
- Use data sampling for initial exploration
- Implement caching for repeated analysis

## Dashboard Features

### Streamlit Dashboard Layout Structure
- **Header**: Title (left) + Date range filter (right) - applies globally to all charts
- **KPI Row**: 4 metric cards with trend indicators
  - Total Revenue, Monthly Growth, Average Order Value, Total Orders
  - Red for negative trends, green for positive trends
  - Two decimal places for all trend indicators
- **Charts Grid**: 2x2 layout with Plotly visualizations
  - **Revenue trend line chart**: 
    - Solid line for current period, dashed line for previous period
    - Grid lines for easier reading
    - Y-axis formatted as $300K instead of $300,000
  - **Top 10 categories bar chart**: 
    - Sorted descending by revenue
    - Blue gradient (lighter for lower values)
    - Values formatted as $300K and $2M
  - **Revenue by state**: US choropleth map with blue gradient color coding
  - **Satisfaction vs delivery time**: Bar chart showing review scores by delivery time buckets
- **Bottom Row**: 2 customer experience cards with uniform heights
  - Average delivery time with trend indicator  
  - Review Score: Large number with stars and "Average Review Score" subtitle

### Technical Features
- **Real-time Filtering**: All visualizations update automatically when year/month is changed
- **Professional Styling**: Business-ready interface with uniform card heights (no icons used)
- **Plotly Charts**: Interactive, publication-quality visualizations
- **Currency Formatting**: Shows $300K instead of $300,000 for better readability
- **Trend Indicators**: Color-coded arrows with precise percentages (2 decimal places)
- **Responsive Design**: Adapts to different screen sizes
- **Error Handling**: Graceful handling of missing data

### Dashboard Usage Instructions

1. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

2. **Use the Filters**:
   - **Year Filter**: Select analysis year (top-right dropdown) 
   - **Month Filter**: Choose specific month or "All Months" for full year analysis
   - All charts update automatically when filters change

3. **Interpret the Metrics**:
   - **Green arrows (↗)**: Positive trends vs previous period
   - **Red arrows (↘)**: Negative trends vs previous period  
   - **KPI Cards**: Show current values with year-over-year comparisons
   - **Charts**: Interactive - hover for detailed values

4. **Navigation Tips**:
   - Charts are fully interactive (zoom, pan, select)
   - Choropleth map shows state-level revenue distribution
   - Delivery time analysis helps identify customer satisfaction patterns

## Future Enhancements

### Planned Features
- Real-time data connections
- Predictive analytics and forecasting
- Customer segmentation analysis
- A/B testing framework
- Automated report scheduling
- Export functionality (PDF reports)

### Extension Ideas
- Integration with business intelligence tools
- API endpoints for metrics access
- Machine learning model integration
- Advanced statistical analysis
- Mobile-responsive improvements

## Contributing

To extend this analysis framework:

1. Follow the existing code structure and documentation patterns
2. Add comprehensive docstrings to new functions
3. Include unit tests for new business logic
4. Update this README with new features

## License

This project is provided as-is for educational and business analysis purposes.

---

**Note**: This framework is designed to be easily maintained and extended for ongoing business intelligence needs. The modular architecture ensures that updates to data sources or metric calculations can be made without affecting the overall analysis structure.