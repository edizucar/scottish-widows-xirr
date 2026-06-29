from datetime import datetime
import os

import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import newton

load_dotenv()

XLSX_FILE = os.getenv("XLSX_FILE")
CURRENT_VALUE = float(os.getenv("CURRENT_VALUE"))
VALUATION_DATE = pd.Timestamp(os.getenv("VALUATION_DATE"))

if not XLSX_FILE:
    raise ValueError("XLSX_FILE is missing from .env")


def clean_money(value):
    """
    Converts values such as:
    £468.75
    -£2.16
    1,234.56
    """
    if pd.isna(value):
        return 0.0

    return float(
        str(value)
        .replace("£", "")
        .replace(",", "")
        .strip()
    )


def build_cashflow(row):
    transaction_type = str(row["Transaction Type"]).strip().lower()
    value = clean_money(row["Value"])

    if transaction_type == "buy":
        return -abs(value)

    if transaction_type == "sell":
        return abs(value)

    return 0.0


def xnpv(rate, cashflows, dates):
    start_date = dates[0]

    return sum(
        cf / (1 + rate) ** ((date - start_date).days / 365.25)
        for cf, date in zip(cashflows, dates)
    )


def xirr(cashflows, dates):
    return newton(
        lambda r: xnpv(r, cashflows, dates),
        x0=0.10,
        maxiter=1000,
    )


def main():
    df = pd.read_excel(XLSX_FILE)

    required_columns = {
        "Date",
        "Transaction Type",
        "Value",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    df = df[df["Transaction Type"].isin(["Buy", "Sell"])].copy()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["CashFlow"] = df.apply(build_cashflow, axis=1)

    df = df.sort_values("Date")

    cashflows = df["CashFlow"].tolist()
    dates = df["Date"].tolist()

    # Add current portfolio value as final inflow
    cashflows.append(CURRENT_VALUE)
    dates.append(VALUATION_DATE)

    annual_return = xirr(cashflows, dates)

    total_contributions = (
        -df.loc[df["CashFlow"] < 0, "CashFlow"].sum()
    )

    print()
    print(f"Portfolio value:      £{CURRENT_VALUE:,.2f}")
    print(f"Total contributions:  £{total_contributions:,.2f}")
    print(f"Annualized return:    {annual_return:.4%}")
    print()


if __name__ == "__main__":
    main()
