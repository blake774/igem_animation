#!/usr/bin/env python3
"""
payload_budget.py -- how big is the DigiActivate construct, really?

The dry lab script currently says:

    "DigiActivate packs a lot into one cell: a guide protein, a transcriptional
     effector, and an E3 ligase binder for every switch. Natural E3 ligases are
     simply too big to fit alongside everything else."

This script tests that sentence against the actual arithmetic, because the
answer changes depending on which "natural E3 ligase" you mean, and for one of
the three the claim does not survive.

Outputs a table, a machine-readable JSON, and an SVG cassette diagram drawn to
scale with the vector packaging limits marked.

  python3 payload_budget.py --switches 3 --effector VP64
"""

from __future__ import annotations

import argparse
import json
import os

# --------------------------------------------------------------------------
# component library
#
# Lengths are residue counts for the human / S. pyogenes sequences the project
# is actually using. Where a module cannot function alone, the obligate
# partners are listed and counted, because a construct has to encode them too.
# --------------------------------------------------------------------------

COMPONENTS = {
    # -- the scaffold ------------------------------------------------------
    "dCas9": dict(
        aa=1368, kind="scaffold",
        note="S. pyogenes Cas9, D10A/H840A. UniProt Q99ZW2 is 1368 aa."),

    # -- transcriptional effectors ----------------------------------------
    "VP64": dict(
        aa=56, kind="effector",
        note="4x VP16 minimal AD (DALDDFDLDML) plus GS linkers."),
    "p65AD": dict(
        aa=185, kind="effector",
        note="RelA activation domain, ~361-551."),
    "VPR": dict(
        aa=522, kind="effector",
        note="VP64-p65-Rta tripartite activator."),

    # -- PROTAC-binding modules -------------------------------------------
    "binder_denovo": dict(
        aa=132, kind="e3_module", partners=[],
        note="RFdiffusionAA design against the 6W74 pocket, contig 120-140."),
    "BIR3_ciap1": dict(
        aa=110, kind="e3_module", partners=[],
        note="Isolated cIAP1 BIR3 domain. Folds and binds SMAC mimetics alone."),
    "cIAP1_full": dict(
        aa=618, kind="e3_module", partners=[],
        note="Full-length BIRC2. Nobody would fuse this, listed for contrast."),
    "VHL": dict(
        aa=213, kind="e3_module", partners=[("EloB", 118), ("EloC", 112)],
        note="pVHL30 needs ElonginB and ElonginC to fold and stay soluble."),
    "CRBN": dict(
        aa=442, kind="e3_module", partners=[("DDB1", 1140)],
        note="Cereblon needs DDB1. This is the 9FJX arrangement."),
    "CRBN_midi": dict(
        aa=380, kind="e3_module", partners=[],
        note="8RQ9 DDB1-independent construct: DDB1-binding bundle deleted, "
             "12 stabilising mutations. Length approximate."),

    # -- glue --------------------------------------------------------------
    "linker": dict(aa=20, kind="linker", note="XTEN-ish flexible linker."),
    "NLS": dict(aa=7, kind="tag", note="SV40 NLS, typically 2 copies."),
    "tag": dict(aa=12, kind="tag", note="epitope tag for detection."),
}

# --------------------------------------------------------------------------
# vector capacity
#
# These are total cargo limits including promoter, polyA and, for AAV, the
# ITRs. The usable coding window is always smaller than the headline number,
# which is the part that trips people up.
# --------------------------------------------------------------------------

VECTORS = {
    "AAV": dict(
        total_kb=4.7, regulatory_kb=1.0,
        note="ITRs + promoter + polyA eat ~1 kb. Single-vector coding room ~3.7 kb."),
    "Lentivirus": dict(
        total_kb=9.0, regulatory_kb=2.5,
        note="Titre falls off steeply past ~8 kb genomic. Coding room ~6.5 kb."),
    "PiggyBac": dict(
        total_kb=100.0, regulatory_kb=2.0,
        note="Effectively unlimited for this construct; not a viral vector."),
    "Plasmid": dict(
        total_kb=100.0, regulatory_kb=2.0,
        note="Transfection only. Fine for HEK293T validation, not for CAR-T."),
}


def resolve(name: str) -> tuple[int, list[str]]:
    """Residue count for a module including any obligate partners."""
    c = COMPONENTS[name]
    total = c["aa"]
    extra = []
    for pname, paa in c.get("partners", []):
        total += paa
        extra.append(f"{pname} ({paa} aa)")
    return total, extra


def build(e3_module: str, n_switches: int, effector: str,
          linkers_per_switch: int = 2, n_nls: int = 2) -> dict:
    """Assemble one construct and price it out."""
    parts = []

    parts.append(("dCas9", COMPONENTS["dCas9"]["aa"], 1))
    parts.append(("NLS", COMPONENTS["NLS"]["aa"], n_nls))
    parts.append(("tag", COMPONENTS["tag"]["aa"], 1))

    e3_aa, partners = resolve(e3_module)
    parts.append((e3_module, e3_aa, n_switches))
    parts.append((effector, COMPONENTS[effector]["aa"], n_switches))
    parts.append(("linker", COMPONENTS["linker"]["aa"],
                  n_switches * linkers_per_switch))

    total_aa = sum(aa * n for _, aa, n in parts)
    coding_bp = total_aa * 3 + 3          # + stop codon

    return dict(
        e3_module=e3_module, effector=effector, switches=n_switches,
        parts=[dict(name=n, aa_each=aa, copies=c, aa_total=aa * c)
               for n, aa, c in parts],
        obligate_partners=partners,
        total_aa=total_aa,
        coding_bp=coding_bp,
        coding_kb=round(coding_bp / 1000, 2),
        note=COMPONENTS[e3_module]["note"],
    )


def fits(construct: dict) -> dict:
    out = {}
    for vname, v in VECTORS.items():
        room = v["total_kb"] - v["regulatory_kb"]
        out[vname] = dict(
            coding_room_kb=round(room, 2),
            fits=construct["coding_kb"] <= room,
            headroom_kb=round(room - construct["coding_kb"], 2),
        )
    return out


# --------------------------------------------------------------------------
# SVG cassette diagram
# --------------------------------------------------------------------------

PALETTE = {
    "dCas9": "#2E6FA8", "NLS": "#9AA7B4", "tag": "#C8CCD2",
    "linker": "#DDE2E8", "VP64": "#5FBF5F", "p65AD": "#5FBF5F",
    "VPR": "#5FBF5F", "binder_denovo": "#F2B441", "BIR3_ciap1": "#E8663C",
    "cIAP1_full": "#E8663C", "VHL": "#E8663C", "CRBN": "#E8663C",
    "CRBN_midi": "#E8663C",
}
LABEL = {
    "dCas9": "dCas9", "NLS": "NLS", "tag": "tag", "linker": "L",
    "VP64": "VP64", "p65AD": "p65", "VPR": "VPR",
    "binder_denovo": "de novo binder", "BIR3_ciap1": "BIR3",
    "cIAP1_full": "cIAP1", "VHL": "VHL+EloBC", "CRBN": "CRBN+DDB1",
    "CRBN_midi": "CRBN-midi",
}


def svg_cassette(constructs: list[dict], path: str, width: int = 1600) -> None:
    """Draw the constructs to a common bp scale with vector limits marked."""
    pad_l, pad_r, pad_t, row_h, gap = 210, 40, 70, 46, 30
    plot_w = width - pad_l - pad_r

    max_kb = max(max(c["coding_kb"] for c in constructs),
                 VECTORS["Lentivirus"]["total_kb"] -
                 VECTORS["Lentivirus"]["regulatory_kb"]) * 1.08
    scale = plot_w / max_kb

    height = pad_t + len(constructs) * (row_h + gap) + 90
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}" '
         f'font-family="Inter, Helvetica, Arial, sans-serif">',
         f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
         f'<text x="{pad_l}" y="30" font-size="20" font-weight="600" '
         f'fill="#1A1A1A">DigiActivate coding sequence, drawn to scale</text>',
         f'<text x="{pad_l}" y="52" font-size="13" fill="#666">'
         f'{constructs[0]["switches"]} PROTAC switches, {LABEL[constructs[0]["effector"]]} '
         f'effector. Bars are coding sequence only.</text>']

    # vector limit lines
    for vname, colour, dash in (("AAV", "#C0392B", "6,4"),
                                ("Lentivirus", "#B7791F", "6,4")):
        v = VECTORS[vname]
        room = v["total_kb"] - v["regulatory_kb"]
        x = pad_l + room * scale
        s.append(f'<line x1="{x:.1f}" y1="{pad_t - 12}" x2="{x:.1f}" '
                 f'y2="{height - 62}" stroke="{colour}" stroke-width="1.6" '
                 f'stroke-dasharray="{dash}"/>')
        s.append(f'<text x="{x + 6:.1f}" y="{pad_t - 18}" font-size="12" '
                 f'fill="{colour}" font-weight="600">{vname} limit '
                 f'({room:.1f} kb)</text>')

    # bars
    y = pad_t
    for c in constructs:
        s.append(f'<text x="{pad_l - 12}" y="{y + 26}" font-size="14" '
                 f'text-anchor="end" fill="#1A1A1A" font-weight="600">'
                 f'{LABEL[c["e3_module"]]}</text>')
        s.append(f'<text x="{pad_l - 12}" y="{y + 42}" font-size="11.5" '
                 f'text-anchor="end" fill="#777">{c["coding_kb"]:.2f} kb  '
                 f'({c["total_aa"]:,} aa)</text>')

        x = pad_l
        for p in c["parts"]:
            for _ in range(p["copies"]):
                w = (p["aa_each"] * 3 / 1000) * scale
                s.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w, 0.7):.1f}" '
                         f'height="{row_h - 12}" fill="{PALETTE[p["name"]]}" '
                         f'stroke="#FFFFFF" stroke-width="0.8"/>')
                if w > 46:
                    s.append(f'<text x="{x + w/2:.1f}" y="{y + 22}" '
                             f'font-size="11" text-anchor="middle" '
                             f'fill="#FFFFFF" font-weight="600">'
                             f'{LABEL[p["name"]]}</text>')
                x += w
        y += row_h + gap

    # axis
    ay = height - 52
    s.append(f'<line x1="{pad_l}" y1="{ay}" x2="{pad_l + max_kb*scale:.1f}" '
             f'y2="{ay}" stroke="#999" stroke-width="1"/>')
    tick = 1 if max_kb < 12 else 2
    kb = 0
    while kb <= max_kb:
        x = pad_l + kb * scale
        s.append(f'<line x1="{x:.1f}" y1="{ay}" x2="{x:.1f}" y2="{ay+5}" '
                 f'stroke="#999"/>')
        s.append(f'<text x="{x:.1f}" y="{ay+20}" font-size="11.5" '
                 f'text-anchor="middle" fill="#666">{kb}</text>')
        kb += tick
    s.append(f'<text x="{pad_l + max_kb*scale/2:.1f}" y="{ay+38}" '
             f'font-size="12" text-anchor="middle" fill="#666">'
             f'coding sequence (kb)</text>')
    s.append("</svg>")

    with open(path, "w") as fh:
        fh.write("\n".join(s))


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--switches", type=int, default=3)
    ap.add_argument("--effector", default="VP64",
                    choices=["VP64", "p65AD", "VPR"])
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    order = ["binder_denovo", "BIR3_ciap1", "CRBN_midi", "VHL",
             "CRBN", "cIAP1_full"]
    built = [build(m, args.switches, args.effector) for m in order]

    print(f"\nDigiActivate payload, {args.switches} switches, "
          f"{args.effector} effector\n")
    print(f"  {'E3 module':<18}{'total aa':>10}{'coding kb':>12}"
          f"{'AAV':>7}{'Lenti':>8}   obligate partners")
    print("  " + "-" * 84)
    results = []
    for c in built:
        f = fits(c)
        results.append(dict(construct=c, fits=f))
        partners = ", ".join(c["obligate_partners"]) or "-"
        print(f"  {LABEL[c['e3_module']]:<18}{c['total_aa']:>10,}"
              f"{c['coding_kb']:>12.2f}"
              f"{'yes' if f['AAV']['fits'] else 'NO':>7}"
              f"{'yes' if f['Lentivirus']['fits'] else 'NO':>8}   {partners}")

    dn = built[0]
    bir3 = built[1]
    delta = dn["coding_kb"] - bir3["coding_kb"]

    print(f"\n  de novo binder vs isolated BIR3: {delta:+.2f} kb "
          f"({dn['total_aa'] - bir3['total_aa']:+,} aa)")
    if delta >= -0.1:
        print("\n  >> The size argument does NOT hold against isolated BIR3.")
        print("     A 110 aa BIR3 domain folds and binds SMAC mimetics on its")
        print("     own, so the de novo binder is not the smaller option. The")
        print("     claim only survives against the obligate-complex recruiters")
        print("     (CRBN+DDB1, VHL+EloBC) and against full-length ligases,")
        print("     which nobody would fuse anyway.")
        print("\n     See README_payload.md for the framing that does hold.")

    svg = os.path.join(args.out_dir, "payload_cassette.svg")
    svg_cassette(built, svg)
    print(f"\n  wrote {svg}")

    js = os.path.join(args.out_dir, "payload_budget.json")
    with open(js, "w") as fh:
        json.dump(dict(switches=args.switches, effector=args.effector,
                       vectors=VECTORS, results=results), fh, indent=2)
    print(f"  wrote {js}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
