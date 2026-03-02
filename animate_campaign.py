#!/usr/bin/env python3
"""
Generate an animated GIF of the best structure from each cycle of a campaign.

Usage:
    python animate_campaign.py outputs/campaign12_tiny_barrel_elite

Requires:
    - PyMOL (headless): /home/judewells/miniconda3/bin/pymol
    - ImageMagick (convert): for stitching PNGs into GIF
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


PYMOL_BIN = "/home/judewells/miniconda3/bin/pymol"


def find_cycle_structures(campaign_dir: Path) -> list[tuple[int, Path]]:
    """Find sequence_0000.cif files and return sorted (cycle_number, path) pairs."""
    results = []
    for cif in sorted(campaign_dir.glob("sequences_cycle_*/sequence_0000.cif")):
        # Extract cycle number from parent dir name: sequences_cycle_3 -> 3
        dirname = cif.parent.name  # e.g. "sequences_cycle_3"
        cycle_num = int(dirname.split("_")[-1])
        results.append((cycle_num, cif))
    results.sort(key=lambda x: x[0])
    return results


def write_pml_script(
    cycles: list[tuple[int, Path]],
    png_dir: Path,
    width: int = 1200,
    height: int = 900,
) -> str:
    """Generate a .pml script that loads, aligns, and renders each cycle."""
    lines = [
        "# Auto-generated PyMOL script for campaign animation",
        "set ray_opaque_background, off",
        f"viewport {width}, {height}",
        "set antialias, 2",
        "set ray_shadows, 1",
        "set ray_trace_mode, 1",
        "set cartoon_fancy_helices, 1",
        "set cartoon_smooth_loops, 1",
        "set spec_reflect, 0.3",
        "",
    ]

    prev_obj = None
    for i, (cycle_num, cif_path) in enumerate(cycles):
        obj_name = f"cycle_{cycle_num}"
        abs_path = str(cif_path.resolve())

        lines.append(f"# --- Cycle {cycle_num} ---")
        lines.append(f'load {abs_path}, {obj_name}')
        lines.append(f"hide everything, {obj_name}")
        lines.append(f"show cartoon, {obj_name}")
        # Color by chain: target (chain B) in orange, binder (everything else) in cyan
        lines.append(f"color cyan, {obj_name} and not chain B")
        lines.append(f"color orange, {obj_name} and chain B")

        if prev_obj is not None:
            lines.append(f"super {obj_name}, {prev_obj}")
            lines.append(f"hide everything, {prev_obj}")
        else:
            # First structure: set the view
            lines.append(f"orient {obj_name}")
            lines.append("zoom")
            # Store the view so all frames use the same camera
            lines.append("get_view")

        if i == 0:
            # Capture the view from the first frame to reuse
            lines.append("stored.view = cmd.get_view()")

        # Restore consistent camera for every frame
        if i > 0:
            lines.append("cmd.set_view(stored.view)")

        # Render with white background
        white_png = png_dir / f"white_{cycle_num:04d}.png"
        lines.append("set ray_opaque_background, on")
        lines.append("bg_color white")
        lines.append(f"png {white_png}, width={width}, height={height}, ray=1")

        # Render with black background
        black_png = png_dir / f"black_{cycle_num:04d}.png"
        lines.append("bg_color black")
        lines.append(f"png {black_png}, width={width}, height={height}, ray=1")

        # Transparent background off for next iteration
        lines.append("")

        prev_obj = obj_name

    # Clean up: hide the last object too
    if prev_obj:
        lines.append(f"hide everything, {prev_obj}")

    lines.append("quit")
    return "\n".join(lines)


def make_gif(png_dir: Path, output_path: Path, prefix: str, delay: int = 50) -> None:
    """Stitch PNGs into an animated GIF using ImageMagick convert."""
    pngs = sorted(png_dir.glob(f"{prefix}_*.png"))
    if not pngs:
        print(f"No PNGs found with prefix '{prefix}', skipping GIF.")
        return
    cmd = [
        "convert",
        "-delay", str(delay),
        "-loop", "0",
        *[str(p) for p in pngs],
        str(output_path),
    ]
    print(f"Creating {output_path} from {len(pngs)} frames...")
    subprocess.run(cmd, check=True)
    print(f"  -> {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate best structures from a campaign using PyMOL."
    )
    parser.add_argument(
        "campaign_dir",
        type=Path,
        help="Path to campaign output directory (e.g. outputs/campaign12_tiny_barrel_elite)",
    )
    parser.add_argument("--width", type=int, default=1200, help="Image width (default 1200)")
    parser.add_argument("--height", type=int, default=900, help="Image height (default 900)")
    parser.add_argument("--delay", type=int, default=50, help="GIF frame delay in centiseconds (default 50)")
    args = parser.parse_args()

    campaign_dir = args.campaign_dir.resolve()
    if not campaign_dir.is_dir():
        print(f"Error: {campaign_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    cycles = find_cycle_structures(campaign_dir)
    if not cycles:
        print(f"Error: no sequences_cycle_*/sequence_0000.cif found in {campaign_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(cycles)} cycle structures (cycles {cycles[0][0]}-{cycles[-1][0]})")

    # Create output directories
    png_dir = campaign_dir / "animation_frames"
    png_dir.mkdir(exist_ok=True)

    # Write PML script
    pml_script = write_pml_script(cycles, png_dir, width=args.width, height=args.height)
    pml_path = campaign_dir / "animate.pml"
    pml_path.write_text(pml_script)
    print(f"PML script written to {pml_path}")

    # Run PyMOL
    print("Running PyMOL (headless)...")
    result = subprocess.run(
        [PYMOL_BIN, "-cq", str(pml_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"PyMOL stderr:\n{result.stderr}", file=sys.stderr)
        print(f"PyMOL stdout:\n{result.stdout}")
        sys.exit(1)
    print("PyMOL rendering complete.")

    # Stitch into GIFs
    make_gif(png_dir, campaign_dir / "animation_white.gif", "white", delay=args.delay)
    make_gif(png_dir, campaign_dir / "animation_black.gif", "black", delay=args.delay)

    print("Done!")


if __name__ == "__main__":
    main()
