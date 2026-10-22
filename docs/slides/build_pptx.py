#!/usr/bin/env python3
"""
Zero-dependency OpenXML (.pptx) packager for Google I/O Keynote Slides (Dark Mode Stage).
Works with /usr/bin/python3 inside the standard sandbox without python-pptx.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import zipfile

EMU_PER_PX = 6350
SLIDE_W = 1920 * EMU_PER_PX
SLIDE_H = 1080 * EMU_PER_PX


def px(val: float) -> int:
    return int(round(val * EMU_PER_PX))


def make_textbox_xml(
    shape_id: int,
    x_px: float,
    y_px: float,
    w_px: float,
    h_px: float,
    text: str,
    font_pt: float,
    bold: bool = False,
    color_hex: str = "F8F9FA",
    align: str = "ctr",
    font_face: str = "Arial",
) -> str:
    b_attr = ' b="1"' if bold else ""
    sz_val = int(round(font_pt * 100))
    escaped = html.escape(text)
    c_hex = color_hex.lstrip("#").upper()
    return f"""
      <p:sp>
        <p:nvSpPr>
          <p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/>
          <p:cNvSpPr txBox="1"/>
          <p:nvPr/>
        </p:nvSpPr>
        <p:spPr>
          <a:xfrm>
            <a:off x="{px(x_px)}" y="{px(y_px)}"/>
            <a:ext cx="{px(w_px)}" cy="{px(h_px)}"/>
          </a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          <a:noFill/>
        </p:spPr>
        <p:txBody>
          <a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="ctr"/>
          <a:lstStyle/>
          <a:p>
            <a:pPr algn="{align}"/>
            <a:r>
              <a:rPr lang="en-US" sz="{sz_val}"{b_attr} dirty="0">
                <a:solidFill><a:srgbClr val="{c_hex}"/></a:solidFill>
                <a:latin typeface="{font_face}"/>
              </a:rPr>
              <a:t>{escaped}</a:t>
            </a:r>
          </a:p>
        </p:txBody>
      </p:sp>"""


def build_pptx_archive(
    out_path: Path,
    gif_paths: list[Path],
    slide_textboxes: list[list[dict]] | None = None,
) -> None:
    num_slides = len(gif_paths)
    if num_slides == 0:
        raise ValueError("No slide GIFs found to package into .pptx")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        slide_overrides = "\n".join(
            f'  <Override PartName="/ppt/slides/slide{i}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
            for i in range(1, num_slides + 1)
        )
        zf.writestr(
            "[Content_Types].xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="gif" ContentType="image/gif"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
{slide_overrides}
</Types>""",
        )

        zf.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>""",
        )

        sld_id_entries = "\n".join(
            f'    <p:sldId id="{255 + i}" r:id="rId{i + 2}"/>'
            for i in range(1, num_slides + 1)
        )
        zf.writestr(
            "ppt/presentation.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
{sld_id_entries}
  </p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}"/>
  <p:notesSz cx="{SLIDE_H}" cy="{SLIDE_W}"/>
</p:presentation>""",
        )

        pres_rels = "\n".join(
            f'  <Relationship Id="rId{i + 2}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
            f'Target="slides/slide{i}.xml"/>'
            for i in range(1, num_slides + 1)
        )
        zf.writestr(
            "ppt/_rels/presentation.xml.rels",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
{pres_rels}
</Relationships>""",
        )

        zf.writestr(
            "ppt/slideMasters/slideMaster1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg>
      <p:bgPr>
        <a:solidFill><a:srgbClr val="000000"/></a:solidFill>
        <a:effectLst/>
      </p:bgPr>
    </p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr/>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="2147483649" r:id="rId1"/>
  </p:sldLayoutIdLst>
</p:sldMaster>""",
        )

        zf.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>""",
        )

        zf.writestr(
            "ppt/slideLayouts/slideLayout1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr/>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>""",
        )

        zf.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>""",
        )

        zf.writestr(
            "ppt/theme/theme1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="GoogleIODarkTheme">
  <a:themeElements>
    <a:clrScheme name="GoogleIODark">
      <a:dk1><a:srgbClr val="F8F9FA"/></a:dk1>
      <a:lt1><a:srgbClr val="000000"/></a:lt1>
      <a:dk2><a:srgbClr val="9AA0A6"/></a:dk2>
      <a:lt2><a:srgbClr val="181B22"/></a:lt2>
      <a:accent1><a:srgbClr val="8AB4F8"/></a:accent1>
      <a:accent2><a:srgbClr val="81C995"/></a:accent2>
      <a:accent3><a:srgbClr val="F28B82"/></a:accent3>
      <a:accent4><a:srgbClr val="FDD663"/></a:accent4>
      <a:accent5><a:srgbClr val="C58AF9"/></a:accent5>
      <a:accent6><a:srgbClr val="8AB4F8"/></a:accent6>
      <a:hlink><a:srgbClr val="8AB4F8"/></a:hlink>
      <a:folHlink><a:srgbClr val="C58AF9"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="GoogleIOFonts">
      <a:majorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>
      <a:minorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="GoogleIOFmt">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>""",
        )

        for idx, gif_path in enumerate(gif_paths, start=1):
            media_name = f"image{idx}.gif"
            zf.write(gif_path, f"ppt/media/{media_name}")

            boxes_xml = ""
            if slide_textboxes and idx - 1 < len(slide_textboxes):
                for b_idx, box in enumerate(slide_textboxes[idx - 1], start=10):
                    boxes_xml += make_textbox_xml(
                        shape_id=b_idx,
                        x_px=box["x"],
                        y_px=box["y"],
                        w_px=box["w"],
                        h_px=box["h"],
                        text=box["text"],
                        font_pt=box.get("size", 20),
                        bold=box.get("bold", False),
                        color_hex=box.get("color", "F8F9FA"),
                        align=box.get("align", "ctr"),
                    )

            zf.writestr(
                f"ppt/slides/slide{idx}.xml",
                f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr/>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="SlideGIF {idx}"/>
          <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="rId2"/>
          <a:stretch><a:fillRect/></a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="{SLIDE_W}" cy="{SLIDE_H}"/>
          </a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>{boxes_xml}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>""",
            )

            zf.writestr(
                f"ppt/slides/_rels/slide{idx}.xml.rels",
                f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media_name}"/>
</Relationships>""",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Google I/O GIF slides into .pptx")
    parser.add_argument("--slides-dir", required=True, help="Directory containing gifs/slide_*.gif")
    parser.add_argument("--title", default="ClauseGuard_AI", help="Prefix for generated .pptx files")
    parser.add_argument("--manifest", default=None, help="Optional JSON manifest for editable text overlays")
    args = parser.parse_args()

    slides_dir = Path(args.slides_dir)
    gif_dir = slides_dir / "gifs"
    all_gifs = sorted(
        p for p in gif_dir.glob("slide_*.gif") if not p.stem.endswith("_bg")
    )
    if not all_gifs:
        raise SystemExit(f"No slide_*.gif files found in {gif_dir}")

    safe_title = args.title.replace(" ", "_")
    main_pptx = slides_dir / f"{safe_title}_GoogleIO_Slides.pptx"
    build_pptx_archive(main_pptx, all_gifs)
    print(f"Created animated deck: {main_pptx} ({len(all_gifs)} slides)")

    bg_gifs = sorted(gif_dir.glob("slide_*_bg.gif"))
    slide_boxes = None
    if args.manifest and Path(args.manifest).exists():
        slide_boxes = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    editable_pptx = slides_dir / f"{safe_title}_GoogleIO_Editable.pptx"
    build_pptx_archive(
        editable_pptx,
        bg_gifs if len(bg_gifs) == len(all_gifs) else all_gifs,
        slide_textboxes=slide_boxes,
    )
    print(f"Created editable deck: {editable_pptx} ({len(all_gifs)} slides)")


if __name__ == "__main__":
    main()
