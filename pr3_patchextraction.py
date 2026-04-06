import argparse
import json
import io
import time
import math
from pathlib import Path

import ee
import requests
from PIL import Image

# basic config
patch_size_px = 64
scale_m = 10
patch_size_m = patch_size_px * scale_m

# aoi coords
aoi_coords = [
    [79.39351040952359, 21.712965810617668],
    [79.4923869098, 21.65171202430774],
    [79.58027713212675, 21.585324414201096],
    [79.72859191254057, 21.615968606177795],
    [79.9098655227434, 21.59553982641719],
    [80.08564599796898, 21.610861647430557],
    [80.23945391401668, 21.64660627902232],
    [80.36030298609201, 21.564891255043094],
    [80.42553387268352, 21.446067049648914],
    [80.42115643095019, 21.435902027877198],
    [80.4133465572518, 21.427015079188095],
    [80.41000210074174, 21.41460969939466],
    [80.40116535473032, 21.406039522669637],
    [80.40979228251182, 21.381841429851505],
    [80.42082172198766, 21.378741110051934],
    [80.44699832497048, 21.37381739683835],
    [80.45030262901516, 21.38046579776302],
    [80.45154726191518, 21.390311090001383],
    [80.45450892177439, 21.3991971203019],
    [80.46227763743636, 21.40552574469403],
    [80.48391003909055, 21.403085606677713],
    [80.54021327877707, 21.390777168563265],
    [80.55118959855639, 21.37383075191717],
    [80.5716220495717, 21.360724228852696],
    [80.57359729799717, 21.349373456105575],
    [80.58793134643855, 21.34697548411873],
    [80.60355081450182, 21.331305306637855],
    [80.62603363375987, 21.323308111405094],
    [80.65143484111901, 21.32874505347229],
    [80.6685997710663, 21.346971309139477],
    [80.68679577505353, 21.376065840378303],
    [80.69813109427753, 21.403880262823375],
    [80.70448050526664, 21.436959935913297],
    [80.72456288082273, 21.453415490856532],
    [80.73366063299687, 21.466035115525585],
    [80.72971234127138, 21.481849722552308],
    [80.72834335828826, 21.517628730771907],
    [80.7349526305617, 21.53822401794982],
    [80.7182148001052, 21.553709536608938],
    [80.72181790101025, 21.57063080572613],
    [80.71469435559561, 21.578294313570538],
    [80.71237677708626, 21.59808921366017],
    [80.72628342533675, 21.6185219478783],
    [80.72353702431793, 21.635114426016596],
    [80.71117569276365, 21.65553615092704],
    [80.7228475814941, 21.68297300774001],
    [80.72696840222865, 21.697649109656854],
    [80.72971663181147, 21.725085753632097],
    [80.73932785839521, 21.744538686541183],
    [80.755809042753, 21.76399059758153],
    [80.77778159777125, 21.81499735863219],
    [80.82172669851185, 22.018842363874334],
    [80.97553457575076, 22.222394500530214],
    [81.09638360115028, 22.461191116789717],
    [80.6239736664878, 22.582971559830742],
    [80.25044024244217, 22.56775487661798],
    [79.80549598151723, 22.344384322057127],
    [79.31660647436897, 21.78949637226263],
    [79.39351040952359, 21.712965810617668],
]

#ee.Authenticate(auth_mode='notebook')
#ee.Initialize(project='practicum-486616')

# ---- helpers ----

gee_proj_name='practicum-486616'  #edit this to use on different device

def init_gee():
    try:
        ee.Initialize(project=gee_proj_name)
        print("gee ok")
    except:
        print("login...")
        ee.Authenticate()
        ee.Initialize(project=gee_proj_name)


def deg_per_km(lat):
    lat_d = 1 / 110.574
    lon_d = 1 / (111.320 * math.cos(math.radians(lat)))
    return lat_d, lon_d


def make_grid(coords, spacing):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    mid_lat = (lat_min + lat_max) / 2

    lat_step, lon_step = deg_per_km(mid_lat)
    lat_step *= spacing
    lon_step *= spacing

    pts = []

    lat = lat_min + lat_step / 2
    while lat < lat_max:
        lon = lon_min + lon_step / 2
        while lon < lon_max:
            pts.append((lon, lat))
            lon += lon_step
        lat += lat_step

    return pts


def filter_loss(pts, aoi, year):
    img = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    y = year - 2000

    mask = (
        img.select("lossyear").gt(0)
        .And(img.select("lossyear").lte(y))
        .And(img.select("treecover2000").gte(30))
    )

    feats = [ee.Feature(ee.Geometry.Point([p[0], p[1]]), {"lon": p[0], "lat": p[1]}) for p in pts]
    fc = ee.FeatureCollection(feats)

    fc = fc.filterBounds(aoi)

    def check(f):
        v = mask.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=f.geometry(),
            scale=30
        ).get("lossyear")
        return f.set("ok", ee.Algorithms.If(v, 1, 0))

    fc = fc.map(check)
    fc = fc.filter(ee.Filter.eq("ok", 1))

    info = fc.toList(fc.size()).getInfo()

    out = []
    for f in info:
        p = f["properties"]
        out.append((p["lon"], p["lat"]))

    return out


def get_s2(aoi, year):
    def mask(img):
        qa = img.select("QA60")
        m = (qa.bitwiseAnd(1 << 10).eq(0)
             .And(qa.bitwiseAnd(1 << 11).eq(0)))
        return img.updateMask(m).divide(10000)

    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask)
        .select(["B4", "B3", "B2"])
        .median()
    )


def get_patch(img, lon, lat, path):
    reg = ee.Geometry.Point([lon, lat]).buffer(patch_size_m / 2).bounds()

    try:
        url = img.getThumbURL({
            "region": reg,
            "dimensions": f"{patch_size_px}x{patch_size_px}",
            "format": "png",
            "bands": ["B4", "B3", "B2"],
            "min": 0,
            "max": 0.3,
        })

        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return False

        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im = im.resize((patch_size_px, patch_size_px))
        im.save(path)

        return True
    except Exception as e:
        print("err:", e)
        return False


# ---- main ----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--grid_spacing_km", type=float, default=1.0)
    p.add_argument("--output_dir", type=str, default="./target_patches")

    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    init_gee()

    aoi = ee.Geometry.Polygon([aoi_coords])

    print("making grid...")
    grid = make_grid(aoi_coords, a.grid_spacing_km)
    print("grid pts:", len(grid))

    print("checking loss...")
    pts = filter_loss(grid, aoi, a.year)
    print("loss pts:", len(pts))

    if len(pts) == 0:
        print("no data")
        return

    print("getting s2...")
    s2 = get_s2(aoi, a.year)

    meta = []
    count = 0

    for i, (lon, lat) in enumerate(pts):
        name = f"p_{i}.png"
        path = out / name

        good = get_patch(s2, lon, lat, path)

        if good:
            meta.append({"file": name, "lon": lon, "lat": lat})
            count += 1

        if i % 10 == 0:
            print(i, "/", len(pts))

        time.sleep(0.3)

    with open(out / "meta.json", "w") as f:
        json.dump(meta, f)

    print("done", count)


if __name__ == "__main__":
    main()