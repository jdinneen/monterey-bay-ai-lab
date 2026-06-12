import io
import pandas as pd
import urllib.request
from ..core import Adapter, _guard_write, _to_parquet_safe

class SurfriderAdapter(Adapter):
    def iter_chunks(self, start=None, end=None):
        for year in range(2000, 2025):
            yield {"key": str(year), "year": year}

    def fetch_chunk(self, chunk):
        year = chunk["year"]
        url = f"https://mmvk4falrj.execute-api.us-west-2.amazonaws.com/v1/annual?year={year}"
        
        req = urllib.request.Request(url, headers={"User-Agent": "mbal-datafetch/0.1"})
        with urllib.request.urlopen(req) as response:
            csv_data = response.read().decode("utf-8")
        
        df = pd.read_csv(io.StringIO(csv_data))
        return self.normalize(df)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=[
                "sample_id", "station", "sample_date", "bacteria_result",
                "bacteria_modifier", "state", "site_id", "site_name",
                "latitude", "longitude", "source",
            ])
        if {"sample_id", "station", "sample_date", "bacteria_result"}.issubset(df.columns):
            return df

        out = pd.DataFrame()
        out["sample_id"] = df.get("sample id", pd.Series(dtype="string")).astype("string")
        site_id = df.get("site id", pd.Series(dtype="string")).astype("string")
        site_name = df.get("site name", pd.Series(dtype="string")).astype("string")
        out["station"] = site_id.str.cat(site_name, sep="|").str.strip("|")
        out["sample_date"] = pd.to_datetime(
            df.get("stored collectionTime"),
            errors="coerce",
            format="mixed",
            utc=True,
        )
        out["bacteria_result"] = pd.to_numeric(df.get("Enterococcus (mpn/100mL)"), errors="coerce")
        out["bacteria_modifier"] = df.get("Enterococcus modifier", pd.Series(dtype="string")).astype("string")
        out["state"] = df.get("state", pd.Series(dtype="string")).astype("string")
        out["site_id"] = site_id
        out["site_name"] = site_name
        out["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
        out["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
        out["source"] = "Surfrider Blue Water Task Force"

        for raw_col, clean_col in [
            ("Ecoli (mpn/100mL)", "ecoli_result"),
            ("Ecoli modifier", "ecoli_modifier"),
            ("Total Coliform (mpn/100mL)", "total_coliform_result"),
            ("Total Coliform modifier", "total_coliform_modifier"),
            ("lab id", "lab_id"),
            ("lab name", "lab_name"),
            ("publication time", "publication_time"),
        ]:
            if raw_col in df.columns:
                out[clean_col] = df[raw_col]

        out = out[out["state"].str.casefold() == "california"].copy()
        out = out.dropna(subset=["sample_id", "sample_date", "bacteria_result"])
        return out

    def _consolidate(self):
        frames = []
        for p in sorted(self.raw_dir.glob("chunk_*.parquet")):
            try:
                d = pd.read_parquet(p)
            except Exception as exc:  # noqa: BLE001
                print(f"[{self.source}] skipping unreadable chunk {p.name}: {exc}")
                continue
            d = self.normalize(d)
            if len(d):
                frames.append(d)

        out_path = self.curated_path
        _guard_write(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            _to_parquet_safe(pd.DataFrame(columns=self.spec.required_columns), out_path)
            return out_path

        df = pd.concat(frames, ignore_index=True)
        if "sample_id" in df.columns:
            df = df.drop_duplicates(["sample_id"])
        df = df.sort_values("sample_date")
        _to_parquet_safe(df, out_path)
        return out_path
