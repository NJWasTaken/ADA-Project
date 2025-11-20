import re
from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np
import pandas as pd


DATA_DIR = Path("data/cross-college-pes-rvce-bms-2025")
OUTPUT_DIR = Path("processed_data")


EXPECTED_HEADERS = [
    "Company",
    "Date of OA",
    "Offline Test",
    "Tier",
    "CTC",
    "Role",
    "Type",
    "Total Offers",
    "CGPA cutoff",
    "Allows ECE",
    "MCA",
    "MTech (CS)",
    "Any comments/questions/topics asked",
]


def detect_header_row_pes(file_path: Path, max_rows: int = 30) -> int:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i in range(max_rows):
            line = f.readline()
            if not line:
                break
            lowered = line.strip().lower()
            if "company" in lowered and "tier" in lowered and "ctc" in lowered:
                return i
    return 0


def parse_bool(x: Any) -> Optional[bool]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"true", "yes"}:
        return True
    if s in {"false", "no"}:
        return False
    return None


def parse_date(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            return None
        if hasattr(dt, "tz") and dt.tz is not None:
            dt = dt.tz_localize(None)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def parse_total_offers(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s == "":
        return None
    m = re.findall(r"\d+", s)
    if not m:
        return None
    try:
        return int(m[-1])
    except Exception:
        return None


def parse_ctc_fields(x: Any) -> Dict[str, Optional[float]]:
    """
    Extracts total CTC LPA, base LPA, and internship stipend monthly INR when present.
    Handles mixed strings like:
    - "32 LPA (19 base, 8 Esops), 80k intern"
    - "17 CTC, 30k Intern"
    - "1L/month Stipend"
    - "1.2 Lakh (intern)"
    """
    res = {"Total_CTC_LPA": None, "Base_LPA": None, "Internship_Stipend_Monthly_INR": None}
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return res
    s = str(x)

    # Total CTC or LPA
    # Prefer numbers followed by LPA/CTC
    m_lpa = re.findall(r"(\d+(?:\.\d+)?)\s*(?:lpa|ctc)", s, flags=re.IGNORECASE)
    if m_lpa:
        try:
            res["Total_CTC_LPA"] = float(m_lpa[0])
        except Exception:
            pass

    # Base
    m_base = re.findall(r"(\d+(?:\.\d+)?)\s*(?:base)", s, flags=re.IGNORECASE)
    if m_base:
        try:
            res["Base_LPA"] = float(m_base[0])
        except Exception:
            pass

    # Internship stipend monthly INR
    # Patterns: "80k intern", "50k Intern", "1L Intern", "1L/month", "1.2 Lakh (intern)"
    stipend = None
    # 1L or 1 Lakh per month
    m_lakh_pm = re.findall(r"(\d+(?:\.\d+)?)\s*(?:lakh|l)\s*(?:/\s*month|month|per\s*month)?", s, flags=re.IGNORECASE)
    if m_lakh_pm:
        try:
            val = float(m_lakh_pm[0]) * 100000.0
            stipend = val
        except Exception:
            pass

    # 80k style
    m_k = re.findall(r"(\d+(?:\.\d+)?)\s*k\b", s, flags=re.IGNORECASE)
    if stipend is None and m_k:
        try:
            stipend = float(m_k[0]) * 1000.0
        except Exception:
            pass

    # Explicit intern keyword after amount (fallback)
    if stipend is None:
        m_amt = re.findall(r"(\d+(?:\.\d+)?)", s)
        if m_amt and re.search(r"intern", s, flags=re.IGNORECASE):
            # Heuristic: if there's an amount and 'intern' present, prefer the last amount if no LPA match
            try:
                val = float(m_amt[-1])
                # If number looks like whole thousands and less than say 500k, treat as INR monthly
                if val <= 200 and re.search(r"lakh|l", s, flags=re.IGNORECASE):
                    stipend = val * 100000.0
                elif val < 500000:
                    stipend = val
            except Exception:
                pass

    res["Internship_Stipend_Monthly_INR"] = stipend
    return res


def clean_pes_csv(in_path: Path, out_path: Path) -> pd.DataFrame:
    header_row = detect_header_row_pes(in_path)
    df = pd.read_csv(in_path, header=header_row, encoding="utf-8", engine="python")
    # Trim columns
    df.columns = [str(c).strip() for c in df.columns]

    # Keep expected columns only when present
    for col in EXPECTED_HEADERS:
        if col not in df.columns:
            df[col] = np.nan

    # Drop rows with no company
    df = df[df["Company"].notna()]
    df = df[df["Company"].astype(str).str.strip() != ""]

    # Normalize fields
    cleaned_records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        company = str(row["Company"]).strip()
        date_oa = parse_date(row["Date of OA"]) if pd.notna(row["Date of OA"]) else None
        offline_test = parse_bool(row["Offline Test"]) if pd.notna(row["Offline Test"]) else None
        tier = str(row["Tier"]).strip() if pd.notna(row["Tier"]) else None
        role = str(row["Role"]).strip() if pd.notna(row["Role"]) else None
        type_ = str(row["Type"]).strip() if pd.notna(row["Type"]) else None
        total_offers = parse_total_offers(row["Total Offers"]) if pd.notna(row["Total Offers"]) else None
        cgpa_cutoff = None
        if pd.notna(row["CGPA cutoff"]):
            try:
                cgpa_cutoff = float(re.findall(r"[\d.]+", str(row["CGPA cutoff"]))[-1])
            except Exception:
                pass
        allows_ece = parse_bool(row["Allows ECE"]) if pd.notna(row["Allows ECE"]) else None
        allows_mca = parse_bool(row["MCA"]) if pd.notna(row["MCA"]) else None
        allows_mtech_cs = parse_bool(row["MTech (CS)"]) if pd.notna(row["MTech (CS)"]) else None
        comments = str(row["Any comments/questions/topics asked"]).strip() if pd.notna(row["Any comments/questions/topics asked"]) else None

        comp = parse_ctc_fields(row["CTC"]) if pd.notna(row["CTC"]) else {"Total_CTC_LPA": None, "Base_LPA": None, "Internship_Stipend_Monthly_INR": None}

        cleaned_records.append(
            {
                "Year": 2025,
                "College": "PES",
                "Company": company,
                "Date_OA": date_oa,
                "Offline_Test": offline_test,
                "Tier": tier,
                "CTC_LPA": comp["Total_CTC_LPA"],
                "Base_LPA": comp["Base_LPA"],
                "Internship_Stipend_Monthly_INR": comp["Internship_Stipend_Monthly_INR"],
                "Role": role,
                "Type": type_,
                "Total_Offers": total_offers,
                "CGPA_Cutoff": cgpa_cutoff,
                "Allows_ECE": allows_ece,
                "Allows_MCA": allows_mca,
                "Allows_MTech_CS": allows_mtech_cs,
                "Comments": comments,
                "Source_File": in_path.name,
            }
        )

    cleaned_df = pd.DataFrame.from_records(cleaned_records,
        columns=[
            "Year","College","Company","Date_OA","Offline_Test","Tier",
            "CTC_LPA","Base_LPA","Internship_Stipend_Monthly_INR","Role","Type",
            "Total_Offers","CGPA_Cutoff","Allows_ECE","Allows_MCA","Allows_MTech_CS",
            "Comments","Source_File"
        ]
    )

    # Coerce numerics
    cleaned_df["CTC_LPA"] = pd.to_numeric(cleaned_df["CTC_LPA"], errors="coerce")
    cleaned_df["Base_LPA"] = pd.to_numeric(cleaned_df["Base_LPA"], errors="coerce")
    cleaned_df["Internship_Stipend_Monthly_INR"] = pd.to_numeric(cleaned_df["Internship_Stipend_Monthly_INR"], errors="coerce")
    cleaned_df["Total_Offers"] = pd.to_numeric(cleaned_df["Total_Offers"], errors="coerce")
    cleaned_df["CGPA_Cutoff"] = pd.to_numeric(cleaned_df["CGPA_Cutoff"], errors="coerce")

    return cleaned_df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pes_path = DATA_DIR / "Cross-College companies info - PES.csv"
    out_path = OUTPUT_DIR / "cross_college_PES_cleaned.csv"
    df = clean_pes_csv(pes_path, out_path)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")
    print("Summary:")
    print(f"  Unique companies: {df['Company'].nunique()}")
    print(f"  Rows with CTC: {df['CTC_LPA'].notna().sum()}")
    print(f"  Rows with stipend: {df['Internship_Stipend_Monthly_INR'].notna().sum()}")


if __name__ == "__main__":
    main()