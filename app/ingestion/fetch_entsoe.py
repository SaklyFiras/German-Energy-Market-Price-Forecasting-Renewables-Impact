# ingestion/fetch_entsoe.py
import os, io
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd, requests
import xml.etree.ElementTree as ET

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)

API   = "https://web-api.tp.entsoe.eu/api"
TOKEN = (os.getenv("ENTSOE_TOKEN") or "").strip()  # trims any \r or spaces
BZN   = "10Y1001A1001A82H"    # DE-LU
DOC   = "A44"                  # Day-ahead prices

def fetch_window(start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    params = {
        "securityToken": TOKEN,
        "documentType": DOC,
        "in_Domain": BZN,
        "out_Domain": BZN,
        "periodStart": start_utc.strftime("%Y%m%d%H%M"),
        "periodEnd":   end_utc.strftime("%Y%m%d%H%M"),
    }
    r = requests.get(API, params=params, timeout=120)
    r.raise_for_status()
    root = ET.parse(io.BytesIO(r.content)).getroot()
    ns = {"ns": root.tag.split('}')[0].strip('{')}
    rows = []
    for ts in root.findall(".//ns:TimeSeries", ns):
        currency = (ts.findtext(".//ns:currency_Unit.name", default="EUR", namespaces=ns) or "EUR")
        for period in ts.findall(".//ns:Period", ns):
            start = pd.to_datetime(period.findtext("ns:timeInterval/ns:start", namespaces=ns), utc=True)
            for p in period.findall("ns:Point", ns):
                pos   = int(p.findtext("ns:position", namespaces=ns))
                price = float(p.findtext("ns:price.amount", namespaces=ns))
                ts_utc = start + timedelta(hours=pos-1)
                rows.append({"ts_utc": ts_utc, "price_eur_mwh": price, "currency": currency})
    return pd.DataFrame(rows).sort_values("ts_utc").drop_duplicates("ts_utc")

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Set ENTSOE_TOKEN in .env and pass it into the 'py' container.")
    end   = pd.Timestamp.utcnow().floor("h").to_pydatetime().replace(tzinfo=timezone.utc)
    start = end - timedelta(days=30)
    df = fetch_window(start, end)
    out = RAW / "entsoe_day_ahead.parquet"
    if out.exists():
        old = pd.read_parquet(out)
        df = pd.concat([old, df]).sort_values("ts_utc").drop_duplicates("ts_utc", keep="last")
    df.to_parquet(out, index=False)
    print("Saved:", out, "rows:", len(df))
