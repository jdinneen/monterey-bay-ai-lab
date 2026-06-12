import pandas as pd

from ops.build_lakehouse import _caloes_county_daily
from ops.data_fetch.adapters.cencoos import (
    CencoosMbalMooringsAdapter,
    CencoosOceanAcidificationAdapter,
)
from ops.data_fetch.registry import REGISTRY


def test_cencoos_registry_uses_real_mbari_erddap_dataset_ids():
    mooring_ids = set(CencoosMbalMooringsAdapter.DATASETS)
    oa_ids = set(CencoosOceanAcidificationAdapter.DATASETS)

    assert "org_mbari_m1" in mooring_ids
    assert "org_mbal_m1" not in mooring_ids
    assert {"oa1-mbari-buoy-1", "oa2-mbari-buoy"}.issubset(oa_ids)
    assert "oa1-mbal-buoy-1" not in oa_ids
    assert "org_mbari_m1" in REGISTRY["cencoos_mbal_moorings"].endpoint


def test_caloes_county_daily_uses_prior_day_events_only(tmp_path):
    path = tmp_path / "caloes_spills.parquet"
    pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                ["2024-01-01T10:00:00Z", "2024-01-03T10:00:00Z"],
                utc=True,
            ),
            "county": ["Monterey County", "Monterey County"],
            "water": ["Yes", "No"],
        }
    ).to_parquet(path, index=False)

    daily = _caloes_county_daily(path).set_index("date").sort_index()

    assert daily.loc[pd.Timestamp("2024-01-01"), "caloes_spill_cnt_7d_county"] == 0.0
    assert daily.loc[pd.Timestamp("2024-01-02"), "caloes_spill_cnt_7d_county"] == 1.0
    assert daily.loc[pd.Timestamp("2024-01-03"), "caloes_spill_cnt_7d_county"] == 1.0
    assert daily.loc[pd.Timestamp("2024-01-03"), "caloes_water_spill_cnt_7d_county"] == 1.0
