#!/usr/bin/env python3
"""Independently rasterize final figure PDFs and audit delivered file formats.

Extra QA dependencies: PyMuPDF, Pillow, resvg-py. Does not read scientific inputs.
Visual review is recorded only when explicitly requested after inspecting previews.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import fitz

os.environ.setdefault("MPLCONFIGDIR", "/tmp/iscci-figure-matplotlib")
import matplotlib
import numpy as np
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--svg-renderer-path",
        type=Path,
        help="Optional isolated resvg-py installation directory; no environment modification",
    )
    parser.add_argument(
        "--record-reviewed",
        action="store_true",
        help="Record visual inspection after the reviewer has inspected current previews",
    )
    args = parser.parse_args()
    if args.svg_renderer_path:
        sys.path.insert(0, str(args.svg_renderer_path.resolve()))
    import resvg_py

    directory = args.directory.resolve()
    manifest = json.loads((directory / "figure_manifest.json").read_text())
    output = directory / "qa"
    output.mkdir(exist_ok=True)
    checks, thumbnails, svg_thumbnails = [], [], []
    for figure in manifest["figures"]:
        name = figure["figure"]
        pdf, svg, png = [
            directory / (name + "." + extension) for extension in ("pdf", "svg", "png")
        ]
        document = fitz.open(pdf)
        if len(document) != 1:
            raise ValueError(f"Expected single readable figure page: {pdf}")
        page = document[0]
        text = page.get_text("dict")
        spans = [
            span
            for block in text["blocks"]
            if "lines" in block
            for line in block["lines"]
            for span in line["spans"]
        ]
        out_of_bounds = [
            span["text"]
            for span in spans
            if not (page.rect + (-1, -1, 1, 1)).contains(fitz.Rect(span["bbox"]))
        ]
        if out_of_bounds:
            raise ValueError(f"PDF text outside page: {name}: {out_of_bounds}")
        root = ET.parse(svg).getroot()
        if not root.tag.endswith("svg"):
            raise ValueError(f"Invalid SVG: {svg}")
        svg_rendered = output / (name + "_svg.png")
        svg_rendered.write_bytes(
            resvg_py.svg_to_bytes(
                svg_path=str(svg),
                width=1400,
                dpi=96.0,
                skip_system_fonts=True,
                font_dirs=[str(Path(matplotlib.get_data_path()) / "fonts/ttf")],
            )
        )
        with Image.open(svg_rendered) as svg_image:
            svg_pixel_sd = np.asarray(svg_image).std()
            if svg_pixel_sd < 5:
                raise ValueError(f"Blank SVG raster: {name}")
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        rendered = output / (name + "_pdf.png")
        pixmap.save(rendered)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8)
        if pixels.std() < 5:
            raise ValueError(f"Blank PDF raster: {name}")
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            if min(dpi) < 299:
                raise ValueError(f"PNG below 300 dpi: {name}: {dpi}")
            png_size = image.size
        preview = Image.open(rendered).convert("RGB")
        preview.thumbnail((1000, 880))
        tile = Image.new("RGB", (1020, 925), "white")
        ImageDraw.Draw(tile).text((10, 5), name, fill="black")
        tile.paste(preview, ((1020 - preview.width) // 2, 30))
        thumbnails.append(tile)
        svg_preview = Image.open(svg_rendered).convert("RGB")
        svg_preview.thumbnail((1000, 880))
        svg_tile = Image.new("RGB", (1020, 925), "white")
        ImageDraw.Draw(svg_tile).text((10, 5), name + " (SVG)", fill="black")
        svg_tile.paste(svg_preview, ((1020 - svg_preview.width) // 2, 30))
        svg_thumbnails.append(svg_tile)
        checks.append(
            dict(
                figure=name,
                file_sha256={p.name: digest(p) for p in (pdf, svg, png)},
                independent_pdf_raster=str(rendered.relative_to(directory)),
                independent_svg_raster=str(svg_rendered.relative_to(directory)),
                nonblank_svg_pixel_sd=float(svg_pixel_sd),
                pdf_text_within_page=True,
                smallest_font_pt=min(s["size"] for s in spans),
                svg_xml_valid=True,
                png_dpi=list(dpi),
                png_pixels=list(png_size),
                nonblank_pdf_pixel_sd=float(pixels.std()),
            )
        )
        document.close()
    contacts = []
    for start in range(0, len(thumbnails), 4):
        sheet = Image.new("RGB", (2040, 1850), "#eeeeee")
        for position, tile in enumerate(thumbnails[start : start + 4]):
            sheet.paste(tile, ((position % 2) * 1020, (position // 2) * 925))
        filename = output / f"contact_{start//4+1:02d}.png"
        sheet.save(filename)
        contacts.append(str(filename.relative_to(directory)))
        svg_sheet = Image.new("RGB", (2040, 1850), "#eeeeee")
        for position, tile in enumerate(svg_thumbnails[start : start + 4]):
            svg_sheet.paste(tile, ((position % 2) * 1020, (position // 2) * 925))
        svg_filename = output / f"svg_contact_{start//4+1:02d}.png"
        svg_sheet.save(svg_filename)
        contacts.append(str(svg_filename.relative_to(directory)))
    record = dict(
        figure_count=len(checks),
        checks=checks,
        contacts=contacts,
        automated_checks_passed=True,
        visually_reviewed=args.record_reviewed,
        review_definition="Independent PDF and resvg SVG rasters and delivered PNGs checked for clipping, legibility, common scales, sparse missing cells, and misleading labels",
        svg_renderer={"name": "resvg-py", "version": resvg_py.__version__},
        figure_manifest_sha256=digest(directory / "figure_manifest.json"),
    )
    (directory / "visual_qa.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"Audited {len(checks)} PDF/SVG/PNG filesets; {len(contacts)} contact sheets in {output}"
    )


if __name__ == "__main__":
    main()
