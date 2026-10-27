# Market Mix Modelling (MMM) App

A Streamlit application for exploring marketing spend, analysing sales performance, and estimating the contribution of different marketing channels using statistical and machine learning techniques.

## Features

- Upload marketing data in CSV format.
- Validate required columns and inspect data quality.
- Visualise sales trends and marketing spend.
- Explore relationships between channel spending and sales.
- Estimate marketing channel contributions using supported modelling methods.

> Note: Actual features depend on the implementation in `app.py`. Model estimates are not necessarily causal effects.

## Prerequisites

- Python 3.10 or later.
- pip (Python package installer).

## Installation

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Create and activate a virtual environment (recommended).

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

From the project root, execute:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, typically `http://localhost:8501`. Open it in your browser and upload your CSV file.

## Input CSV Schema

The uploaded CSV must contain the following columns:

| Column | Description | Expected format |
|---|---|---|
| `date` | Date of the observation | Date, e.g. `2026-01-01` |
| `Sales` | Sales outcome for the period | Numeric |
| `TV_Spend` | TV advertising expenditure | Numeric |
| `Radio_Spend` | Radio advertising expenditure | Numeric |
| `Digital_Spend` | Digital advertising expenditure | Numeric |

### Example

```csv
date,Sales,TV_Spend,Radio_Spend,Digital_Spend
2026-01-01,125000,12000,5000,8000
2026-01-08,132000,14000,4500,9000
2026-01-15,128000,11000,6000,8500
2026-01-22,145000,16000,7000,11000
```

### Data Guidelines

- Use one row per consistent time period (daily, weekly, or monthly).
- Ensure `date` contains valid dates and is consistently formatted.
- Ensure sales and spend columns contain numeric values.
- Avoid missing values, duplicate dates, and invalid or negative values unless explicitly supported by the application.
- Use the same currency and units throughout the dataset.
- Provide sufficient historical observations for reliable modelling.

## Screenshots

Add application screenshots to a `screenshots/` directory and update the paths below.

### Application Dashboard

![Application Dashboard](screenshots/dashboard.png)

### Sales and Marketing Spend Analysis

![Sales and Marketing Spend Analysis](screenshots/analysis.png)

### Model Results and Channel Contributions

![Model Results and Channel Contributions](screenshots/model-results.png)

## Project Structure

```text
mmm-app/
├── app.py
├── requirements.txt
├── README.md
└── screenshots/
    ├── dashboard.png
    ├── analysis.png
    └── model-results.png
```

## Troubleshooting

- **Missing dependency:** Run `pip install -r requirements.txt`.
- **Invalid CSV:** Check column names, date formats, and numeric values.
- **Application fails to start:** Verify your Python version and run the command from the project directory.
- **Port already in use:** Run `streamlit run app.py --server.port 8502`.

## Disclaimer

MMM results depend on data quality, model assumptions, seasonality, lagged effects, and other relevant factors. Channel contribution estimates should not automatically be interpreted as causal effects or guaranteed future performance.