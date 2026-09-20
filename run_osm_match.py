"""Match each heat-source cluster against OpenStreetMap land-use features.
Run from the sih-fire folder:  python run_osm_match.py
Safe to stop (Ctrl+C) and re-run: it resumes where it left off."""
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox

CLUSTERS = "outputs/clusters.csv"
CELLS = "outputs/site_cells.csv"
OUT = "outputs/osm_features.csv"

ox.settings.requests_timeout = 120
ox.settings.use_cache = True
ox.settings.cache_folder = "cache"
ox.settings.log_console = False

TAGS = {
    "landuse": ["industrial", "quarry", "landfill", "forest"],
    "natural": ["wood"],
    "man_made": ["works", "chimney", "flare"],
    "power": ["plant", "generator"],
}
CATS = ["industrial", "mining", "landfill", "power_gen", "forest"]


def col(g, name):
    if name in g.columns:
        return g[name]
    return pd.Series(np.nan, index=g.index, dtype=object)


def categorize(g):
    lu, nat = col(g, "landuse"), col(g, "natural")
    mm, pw = col(g, "man_made"), col(g, "power")
    cat = pd.Series(np.nan, index=g.index, dtype=object)
    cat[lu.eq("forest") | nat.eq("wood")] = "forest"
    cat[pw.eq("generator")] = "power_gen"
    cat[lu.eq("landfill")] = "landfill"
    cat[lu.eq("quarry")] = "mining"
    cat[lu.eq("industrial") | mm.isin(["works", "chimney", "flare"]) | pw.eq("plant")] = "industrial"
    return cat


def process(row, cells):
    lat, lon = row["lat"], row["lon"]
    radius = int(min(6000, max(1500, row["extent_km"] * 500 + 1500)))
    out = {"cluster": int(row["cluster"]), "radius_m": radius}

    try:
        feats = ox.features_from_point((lat, lon), tags=TAGS, dist=radius)
    except Exception as e:
        if type(e).__name__ == "InsufficientResponseError":
            out.update(n_features=0, status="no_features")
            return out
        raise

    feats = feats[feats.geometry.notna()].copy()
    feats["cat"] = categorize(feats)
    feats = feats[feats["cat"].notna()]
    out["n_features"] = len(feats)
    if feats.empty:
        out["status"] = "no_features"
        return out

    utm = feats.estimate_utm_crs()
    fm = feats.to_crs(utm)
    pts = gpd.GeoSeries(
        gpd.points_from_xy(list(cells["lon"]) + [lon], list(cells["lat"]) + [lat]),
        crs="EPSG:4326",
    ).to_crs(utm)  # last point = cluster centre

    for c in CATS:
        sub = fm[fm["cat"] == c]
        out[f"n_{c}"] = len(sub)
        if sub.empty:
            out[f"{c}_dist_center_m"] = np.nan
            out[f"{c}_min_dist_m"] = np.nan
            out[f"{c}_frac_cells_inside"] = 0.0
        else:
            d = np.min([pts.distance(g).values for g in sub.geometry], axis=0)
            out[f"{c}_dist_center_m"] = round(float(d[-1]))
            out[f"{c}_min_dist_m"] = round(float(d[:-1].min()))
            out[f"{c}_frac_cells_inside"] = round(float((d[:-1] == 0).mean()), 3)
    out["status"] = "ok"
    return out


def main():
    clusters = pd.read_csv(CLUSTERS).sort_values("cluster_days", ascending=False)
    cells_all = pd.read_csv(CELLS)

    records = []
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        old = old[old["status"].isin(["ok", "no_features"])]
        records = old.to_dict("records")
    done = {int(r["cluster"]) for r in records}

    todo = clusters[~clusters["cluster"].isin(done)]
    print(f"{len(done)} already done, {len(todo)} to go", flush=True)

    t_start = time.time()
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        cells = cells_all[cells_all["cluster"] == row["cluster"]]
        t0 = time.time()
        rec = None
        for attempt in range(3):
            try:
                rec = process(row, cells)
                break
            except Exception as e:
                print(f"  cluster {int(row['cluster'])} attempt {attempt + 1} failed: {str(e)[:80]}", flush=True)
                time.sleep(20)
        if rec is None:
            rec = {"cluster": int(row["cluster"]), "status": "error"}
        records.append(rec)
        pd.DataFrame(records).to_csv(OUT, index=False)

        avg = (time.time() - t_start) / i
        left = avg * (len(todo) - i) / 60
        print(f"{i}/{len(todo)} cluster {int(row['cluster'])} "
              f"{rec['status']} in {time.time() - t0:.0f}s (~{left:.0f} min left)", flush=True)
        time.sleep(2)

    print("Finished. Saved to", OUT)


if __name__ == "__main__":
    main()