import os
import json
import pandas as pd


def build_timeline(df: pd.DataFrame) -> pd.DataFrame:
    # Parse date with day-first format; keep original column as-is
    df = df.copy()
    df["Date_OA_parsed"] = pd.to_datetime(df["Date_OA"], dayfirst=True, errors="coerce")
    # Keep only rows with a valid date
    timeline_df = df.dropna(subset=["Date_OA_parsed"]) \
        [["Date_OA_parsed", "Date_OA", "Company", "Role", "Type", "Tier", "Total_Offers"]]
    # Sort chronologically
    timeline_df = timeline_df.sort_values("Date_OA_parsed").reset_index(drop=True)
    return timeline_df


def summarize_timeline(timeline_df: pd.DataFrame) -> dict:
    summary = {}
    if len(timeline_df) == 0:
        return {
            "entries": 0,
            "unique_companies": 0,
            "earliest_date": None,
            "latest_date": None,
            "top_companies_by_frequency": [],
        }
    summary["entries"] = int(len(timeline_df))
    summary["unique_companies"] = int(timeline_df["Company"].nunique())
    summary["earliest_date"] = timeline_df["Date_OA_parsed"].min().date().isoformat()
    summary["latest_date"] = timeline_df["Date_OA_parsed"].max().date().isoformat()

    # Company frequency
    freq = (
        timeline_df["Company"].value_counts().rename_axis("Company").reset_index(name="count")
    )
    summary["top_companies_by_frequency"] = (
        freq.sort_values("count", ascending=False).head(15).to_dict(orient="records")
    )

    # First appearance per company
    firsts = (
        timeline_df.groupby("Company")["Date_OA_parsed"].min().reset_index()
    )
    firsts["first_appearance_date"] = firsts["Date_OA_parsed"].dt.date.astype(str)
    summary["company_first_appearance"] = firsts[["Company", "first_appearance_date"]] \
        .sort_values("first_appearance_date") \
        .to_dict(orient="records")

    return summary


def group_by_date(timeline_df: pd.DataFrame) -> list:
    # Group companies by exact date
    grouped = (
        timeline_df.assign(date_iso=timeline_df["Date_OA_parsed"].dt.date.astype(str))
        .groupby("date_iso")
    )
    result = []
    for date_iso, grp in grouped:
        companies = grp["Company"].tolist()
        roles = grp[["Company", "Role", "Type", "Tier", "Total_Offers"]] \
            .to_dict(orient="records")
        result.append({
            "date": date_iso,
            "companies": companies,
            "details": roles,
            "count": int(len(grp))
        })
    # Sort by date ascending
    result.sort(key=lambda x: x["date"])
    return result


def main():
    src_path = r"c:\\Users\\Prateek\\OneDrive\\Dokumen\\SEM5\\ADA\\ADA-Project\\new_clean_data\\cross_college_PES_cleaned.csv"
    out_dir = r"c:\\Users\\Prateek\\OneDrive\\Dokumen\\SEM5\\ADA\\ADA-Project\\analysis_outputs\\cross_college"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(src_path)
    timeline_df = build_timeline(df)

    # Save CSV timeline
    timeline_csv = os.path.join(out_dir, "company_appearance_timeline.csv")
    timeline_df.to_csv(timeline_csv, index=False)

    # Build JSON outputs
    summary = summarize_timeline(timeline_df)
    grouped_by_date = group_by_date(timeline_df)

    json_path = os.path.join(out_dir, "company_appearance_timeline.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": src_path,
            "summary": summary,
            "timeline": grouped_by_date,
        }, f, ensure_ascii=False, indent=2)

    # Print a concise summary to stdout
    print(f"Timeline entries: {summary['entries']}")
    print(f"Unique companies: {summary['unique_companies']}")
    print(f"Date range: {summary['earliest_date']} → {summary['latest_date']}")
    print("Top companies by frequency (top 10):")
    for row in summary["top_companies_by_frequency"][:10]:
        print(f"  - {row['Company']}: {row['count']}")

    # Show first 10 timeline rows
    head = timeline_df.head(10)
    print("\nFirst 10 appearances:")
    for _, r in head.iterrows():
        date_str = r["Date_OA_parsed"].date().isoformat()
        print(f"  {date_str} | {r['Company']} | {r['Role']} | {r['Type']} | {r['Tier']}")


if __name__ == "__main__":
    main()