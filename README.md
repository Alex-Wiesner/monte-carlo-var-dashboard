# monte-carlo-var-dashboard

Fast, interactive dashboard provinding risk analytics for a custom stock basket.

**Live demo:** https://monte-carlo-var-dashboard.streamlit.app/

## Installation

```bash
git clone https://github.com/Alex-Wiesner/monte-carlo-var-dashboard
cd monte-carlo-var-dashboard
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## How it works

The Monte Carlo engine works by using geometric Brownian motion with Cholesky-decomposed covariance matrix to simulate thousands of price paths for each ticker, then computes Value-at-Risk (VaR) and Expected Shortfall (ES) at 95% and 99% confidence levels.

## License

MIT License @ 2025 Alex R. Wiesner
