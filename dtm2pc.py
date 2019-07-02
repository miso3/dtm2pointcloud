#!/usr/bin/env python3
from enum import Enum
from pathlib import Path
import argparse

from osgeo import gdal, gdalconst
from plyfile import PlyElement, PlyData
import numpy as np
import osr
import ogr
import pyproj


class System(Enum):
    """
    """

    latlong = "latlong"  # latitude & longitude (deg)
    geocent = "geocent"  # geo centric coordinate (m)
    src = "src"  # inherit source dtm coodinate system
    imgcent = "imgcent"  # image center xyz (m)

    def __str__(self):
        return self.value


def get_args():
    parser = argparse.ArgumentParser(
        "Convert DTM tiff file to point cloud format(PLY)")
    parser.add_argument("dtm", type=Path, help="DTM tiff file")
    parser.add_argument("dest", type=Path, help="PLY file")
    parser.add_argument("--texture", type=Path, help="texture tiff file")
    parser.add_argument("--ascii", default=False, action="store_true",
                        help="save as ascii format")
    parser.add_argument("--system", type=System, choices=list(System),
                        default=System.src,
                        help="choose output coordinate system")
    parser.add_argument("--downsample", type=int, default=1,
                        help="decrease sample rate")

    args = parser.parse_args()

    return args


def to_geocoord_func(geo_trans):
    def inner(p, l):
        x = geo_trans[0] + p * geo_trans[1] + l * geo_trans[2]
        y = geo_trans[3] + p * geo_trans[4] + l * geo_trans[5]
        return x, y

    return inner


def get_map_coord(dtm):
    to_geocoord = to_geocoord_func(dtm.GetGeoTransform())

    idx_x, idx_y = np.meshgrid(range(dtm.RasterXSize), range(dtm.RasterYSize))
    geo_x, geo_y = to_geocoord(idx_x, idx_y)
    elev = dtm.GetRasterBand(1).ReadAsArray().astype(np.float32)
    elev[elev == dtm.GetRasterBand(1).GetNoDataValue()] = np.nan
    return geo_x, geo_y, elev


def transform_coord(system, dtm, map_x, map_y, map_elev):
    if system == System.src:
        return map_x, map_y, map_elev
    elif system == System.imgcent:
        cy, cx = map_x.shape[0] // 2, map_x.shape[1] // 2
        map_x -= map_x[cy, cx]
        map_y -= map_y[cy, cx]
        map_elev -= map_elev[cy, cx]
        return map_x, map_y, map_elev

    dtm_srs = osr.SpatialReference()
    dtm_srs.ImportFromWkt(dtm.GetProjection())
    from_p = pyproj.Proj(dtm_srs.ExportToProj4())

    to_p = pyproj.Proj(
        proj=str(system),
        a=dtm_srs.GetSemiMajor(),
        b=dtm_srs.GetSemiMinor(),
        units="m")

    return pyproj.transform(
        from_p, to_p, map_x, map_y, map_elev, radians=False)


def main():
    args = get_args()

    dtm = gdal.Open(str(args.dtm), gdalconst.GA_ReadOnly)
    map_x, map_y, map_elev = get_map_coord(dtm)
    o_x, o_y, o_z = transform_coord(args.system, dtm, map_x, map_y, map_elev)
    print(o_x.shape)
    if args.downsample > 1:

        o_x = o_x[::args.downsample, ::args.downsample]
        o_y = o_y[::args.downsample, ::args.downsample]
        o_z = o_z[::args.downsample, ::args.downsample]
        map_elev = map_elev[::args.downsample, ::args.downsample]

    body = [o_x, o_y, o_z]
    names = "x, y, z"
    formats = "f4, f4, f4"
    if args.texture:
        # TODO DTMとテクスチャのサイズ|分解能が違う場合に対応する
        tex = gdal.Open(str(args.texture), gdalconst.GA_ReadOnly)
        tex_data = tex.GetRasterBand(1).ReadAsArray().astype(np.uint8)
        if args.downsample > 1:
            tex_data = tex_data[::args.downsample, ::args.downsample]

        body.extend([tex_data, tex_data, tex_data])
        names += ", red, green, blue"
        formats += ", u1, u1, u1"

    # 頂点配列
    vertices = np.core.records.fromarrays(
        np.dstack(body).reshape((-1, len(body))).transpose(),
        names=names,
        formats=formats).flatten()

    # nodataを除去して保存
    valid_vertices = np.isfinite(map_elev.flatten())
    ply_data_elm = [
        PlyElement.describe(
            vertices[valid_vertices], "vertex", comments=["vertices"])
    ]
    ply_data = PlyData(ply_data_elm, text=args.ascii)
    ply_data.write(args.dest)


if __name__ == "__main__":
    main()
