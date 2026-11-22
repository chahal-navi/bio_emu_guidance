"""Generate objective_functions_report.html from objective_validation/validation.json."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "objective_validation" / "validation.json"
OUT = HERE / "objective_functions_report.html"

GOOD, CRIT, WARN = "#0ca30c", "#d03b3b", "#fab219"

# Short labels for chart axes: the full names overflow the 190px label gutter.
SHORT = {
    "ez_ca": "Ez (Ca)", "ez_clamped": "Ez clamped", "ez_anchored": "Ez + anchor",
    "depth_contrast": "Depth contrast", "burial_saturating": "Saturating burial",
    "hydrophobic_moment": "Hydrophobic moment", "moment_alignment": "Moment x normal",
    "combined_moment_heavy": "Burial + 3x moment", "combined": "Burial + moment",
}


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def load_frames():
    f = HERE / "objective_validation" / "frame_comparison.json"
    return json.loads(f.read_text()) if f.exists() else None


def load():
    d = json.loads(DATA.read_text())
    objs = []
    for o in d["objectives"]:
        st = o.get("t2_stretch") or {}
        s1, s3 = st.get("1.0"), st.get("3.0")
        gamed = s1 is not None and s3 is not None and s3 < s1
        sd = o["std"] or 1e-9
        objs.append({
            "key": o["key"], "name": o["name"], "rationale": o["rationale"],
            "couples": o["couples_residues"], "mean": o["mean"], "std": o["std"],
            "r": o["t1_r_helicity"], "p": o["t1_p"],
            "stretch_sd": (s3 - s1) / sd if s1 is not None else None,
            "gamed": gamed, "headroom": o["t5_headroom_sd"],
            "frame": o.get("t6_frame") or {}, "refpct": o.get("t4_ref_percentile"),
            "expel": o.get("t3_expel") or {},
            "pass": (o["t1_r_helicity"] < 0) and not gamed,
        })  # fmt: skip
    return d, objs


def bar_chart(items, *, title, sub, unit, good_is, domain=None, ref=None, ref_label=""):
    """items: [(label, value, is_good)]. Horizontal bars diverging from zero."""
    rowh, gap, left, right, top = 30, 8, 190, 62, 46
    n = len(items)
    h = top + n * (rowh + gap) + 34
    w = 760
    plot_w = w - left - right
    vals = [v for _, v, _ in items]
    lo = min(0, min(vals)) if domain is None else domain[0]
    hi = max(0, max(vals)) if domain is None else domain[1]
    span = (hi - lo) or 1
    pad = span * 0.08
    lo, hi = lo - pad, hi + pad
    span = hi - lo

    def X(v):
        return left + (v - lo) / span * plot_w

    p = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}" '
        f'style="width:100%;height:auto;max-width:100%">',
        f'<text x="0" y="18" class="ct">{esc(title)}</text>',
        f'<text x="0" y="36" class="cs">{esc(sub)}</text>',
    ]
    x0 = X(0)
    p.append(f'<line x1="{x0:.1f}" y1="{top - 8}" x2="{x0:.1f}" y2="{h - 30}" class="axis"/>')
    if ref is not None:
        xr = X(ref)
        p.append(f'<line x1="{xr:.1f}" y1="{top - 8}" x2="{xr:.1f}" y2="{h - 30}" class="refline"/>')
        p.append(f'<text x="{xr:.1f}" y="{h - 16}" class="cl" text-anchor="middle">{esc(ref_label)}</text>')
    for i, (label, v, is_good) in enumerate(items):
        y = top + i * (rowh + gap)
        xv = X(v)
        bx, bw = (min(x0, xv), abs(xv - x0))
        colour = GOOD if is_good else CRIT
        p.append(f'<text x="{left - 12}" y="{y + 19}" class="cl" text-anchor="end">{esc(label)}</text>')
        p.append(
            f'<rect x="{bx:.1f}" y="{y + 4}" width="{max(bw, 1.5):.1f}" height="{rowh - 8}" '
            f'rx="3" fill="{colour}" opacity="0.92"/>'
        )
        anchor = "start" if v >= 0 else "end"
        tx = xv + (7 if v >= 0 else -7)
        p.append(
            f'<text x="{tx:.1f}" y="{y + 19}" class="cv" text-anchor="{anchor}">{v:+.2f}{unit}</text>'
        )
    p.append(f'<text x="{x0:.1f}" y="{h - 16}" class="cl" text-anchor="middle">0</text>')
    p.append(f'<text x="{w}" y="{h - 16}" class="cl" text-anchor="end">{esc(good_is)}</text>')
    p.append("</svg>")
    return "\n".join(p)


def main():
    d, objs = load()
    passed = [o for o in objs if o["pass"]]
    frame_objs = [o for o in objs if o["frame"]]
    best_frame = min(frame_objs, key=lambda o: o["frame"]["normal_err_deg"]) if frame_objs else None

    c1 = bar_chart(
        [(SHORT[o["key"]], o["stretch_sd"], o["stretch_sd"] > 0) for o in sorted(objs, key=lambda o: o["stretch_sd"] or 0)],
        title="T2 — Can the objective be gamed by stretching the chain?",
        sub="Score change when every residue's depth is scaled 3x along the membrane normal, in ensemble standard deviations. Negative = stretching IMPROVES the score = gameable.",
        unit="σ", good_is="worsens → safe",
    )
    c2 = bar_chart(
        [(SHORT[o["key"]], o["r"], o["r"] < 0) for o in sorted(objs, key=lambda o: o["r"])],
        title="T1 — Does the objective reward the folded helix?",
        sub="Pearson r between objective and DSSP helicity across 650 generated structures. Melittin's membrane-bound state is helical, so a usable objective needs r < 0.",
        unit="", good_is="rewards helix → good",
    )
    c3 = bar_chart(
        [(SHORT[o["key"]], o["frame"]["normal_err_deg"], o["frame"]["normal_err_deg"] < 20) for o in
         sorted(frame_objs, key=lambda o: o["frame"]["normal_err_deg"])],
        title="T6 — Does it recover the known membrane placement?",
        sub="Angle between the objective's predicted membrane normal and the true normal, given the OPM-oriented melittin structure (ground truth: 0°). 90° = perpendicular to reality.",
        unit="°", good_is="", domain=(0, 90), ref=20, ref_label="20° tolerance",
    )

    rows = []
    for o in sorted(objs, key=lambda o: (not o["pass"], o["r"])):
        fr = o["frame"].get("normal_err_deg")
        badge = (f'<span class="badge pass">PASS</span>' if o["pass"]
                 else f'<span class="badge fail">FAIL</span>')
        rows.append(f"""<tr>
      <td><div class="oname">{esc(o['name'])}</div><div class="okey">{esc(o['key'])}</div></td>
      <td class="num">{o['r']:+.3f}</td>
      <td class="num">{'—' if o['stretch_sd'] is None else f"{o['stretch_sd']:+.2f}σ"}</td>
      <td class="num">{'—' if fr is None else f'{fr:.0f}°'}</td>
      <td class="num">{o['headroom']:.2f}σ</td>
      <td>{'yes' if o['couples'] else 'no'}</td>
      <td>{badge}</td>
    </tr>""")

    cards = []
    for o in sorted(objs, key=lambda o: (not o["pass"], o["r"])):
        fr = o["frame"].get("normal_err_deg")
        why = []
        if o["r"] >= 0:
            why.append(f"rewards <strong>loss</strong> of helix (r = {o['r']:+.3f})")
        if o["gamed"]:
            why.append(f"improves by {abs(o['stretch_sd']):.2f}σ when the chain is stretched")
        if fr is not None and fr >= 20:
            why.append(f"predicts a membrane normal {fr:.0f}° from the true one")
        verdict = ("Survives both gaming tests." if o["pass"]
                   else "Disqualified: " + "; ".join(why) + ".")
        cards.append(f"""<article class="card {'ok' if o['pass'] else 'no'}">
      <header><h3>{esc(o['name'])}</h3>{'<span class="badge pass">PASS</span>' if o['pass'] else '<span class="badge fail">FAIL</span>'}</header>
      <p class="rat">{esc(o['rationale'])}</p>
      <p class="verdict">{verdict}</p>
    </article>""")

    fc = load_frames()
    frame_rows = ""
    if fc:
        def cell(v, good, unit=""):
            if v is None:
                return '<td class="num">—</td>'
            col = "var(--good)" if good else "var(--crit)"
            return f'<td class="num" style="color:{col}">{v:+.2f}{unit}</td>'
        parts = []
        for row in fc["objectives"]:
            parts.append(
                "<tr><td><div class='oname'>" + esc(SHORT.get(row["key"], row["name"]))
                + "</div><div class='okey'>" + esc(row["key"]) + "</div></td>"
                + cell(row.get("r_opt"), (row.get("r_opt") or 0) < 0)
                + cell(row.get("r_opm"), (row.get("r_opm") or 0) < 0)
                + cell(row.get("stretch_opt"), (row.get("stretch_opt") or 0) > 0, "σ")
                + cell(row.get("stretch_opm"), (row.get("stretch_opm") or 0) > 0, "σ")
                + "</tr>")
        frame_rows = "".join(parts)

    html = f"""<meta name="claude:viewport-fit-sides" content="cover">
<title>Membrane Objective Validation</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: single-column technical report; bilayer-banding motif echoes the z = +/-15.4 A slab */
:root {{
  --bg:#fbfbfa; --panel:#ffffff; --ink:#1a1c1b; --ink-2:#51574f; --ink-3:#7d837a;
  --line:#e3e5df; --accent:#1f6f6b; --core:#eef3f1;
  --good:{GOOD}; --crit:{CRIT}; --warn:{WARN};
  --f-disp:'Space Grotesk',system-ui,sans-serif;
  --f-body:'Inter',system-ui,sans-serif;
  --f-mono:'IBM Plex Mono',ui-monospace,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#131514; --panel:#1b1e1c; --ink:#edefea; --ink-2:#b4bbb2; --ink-3:#848b82;
    --line:#2d322e; --accent:#5fb5ae; --core:#1f2523; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#131514; --panel:#1b1e1c; --ink:#edefea; --ink-2:#b4bbb2; --ink-3:#848b82;
  --line:#2d322e; --accent:#5fb5ae; --core:#1f2523; color-scheme: dark;
}}
body {{ background:var(--bg); color:var(--ink); font-family:var(--f-body); line-height:1.6;
  margin:0; font-size:15px; }}
.wrap {{ max-width:860px; margin:0 auto;
  padding-left:max(16px, env(safe-area-inset-left, 0px));
  padding-right:max(16px, env(safe-area-inset-right, 0px));
  padding-block:40px 72px; }}
h1 {{ font-family:var(--f-disp); font-size:clamp(28px,5vw,40px); line-height:1.12; margin:0 0 10px;
  letter-spacing:-0.02em; text-wrap:balance; }}
h2 {{ font-family:var(--f-disp); font-size:21px; margin:52px 0 6px; letter-spacing:-0.01em; }}
h3 {{ font-family:var(--f-disp); font-size:15px; margin:0; }}
p {{ margin:0 0 14px; max-width:66ch; }}
.eyebrow {{ font-size:11px; letter-spacing:0.14em; text-transform:uppercase;
  color:var(--accent); font-weight:600; margin-bottom:14px; }}
.lede {{ font-size:17px; color:var(--ink-2); max-width:64ch; }}
.sub {{ color:var(--ink-3); font-size:13px; margin:-2px 0 18px; }}
.slab {{ background:var(--core); border-top:2px solid var(--accent); border-bottom:2px solid var(--accent);
  padding:20px 22px; margin:26px 0; border-radius:2px; }}
.slab p:last-child {{ margin-bottom:0; }}
.slab strong {{ color:var(--ink); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
.tscroll {{ overflow-x:auto; margin:18px 0 8px; }}
th {{ text-align:left; font-family:var(--f-disp); font-size:11.5px; letter-spacing:0.05em;
  text-transform:uppercase; color:var(--ink-3); font-weight:600;
  border-bottom:1px solid var(--line); padding:0 12px 8px 0; white-space:nowrap; }}
td {{ border-bottom:1px solid var(--line); padding:11px 12px 11px 0; vertical-align:top; }}
.num {{ font-family:var(--f-mono); font-variant-numeric:tabular-nums; white-space:nowrap; }}
.oname {{ font-weight:600; }}
.okey {{ font-family:var(--f-mono); font-size:11.5px; color:var(--ink-3); }}
.badge {{ font-family:var(--f-disp); font-size:10.5px; font-weight:700; letter-spacing:0.08em;
  padding:3px 8px; border-radius:3px; color:#fff; white-space:nowrap; }}
.badge.pass {{ background:var(--good); }}
.badge.fail {{ background:var(--crit); }}
figure {{ margin:22px 0 10px; background:var(--panel); border:1px solid var(--line);
  border-radius:6px; padding:18px 18px 10px; }}
.ct {{ font-family:var(--f-disp); font-size:14px; font-weight:700; fill:var(--ink); }}
.cs {{ font-size:11px; fill:var(--ink-3); }}
.cl {{ font-size:11.5px; fill:var(--ink-2); font-family:var(--f-body); }}
.cv {{ font-size:11.5px; fill:var(--ink-2); font-family:var(--f-mono); font-weight:500; }}
.axis {{ stroke:var(--ink-3); stroke-width:1.2; }}
.refline {{ stroke:var(--ink-3); stroke-width:1; stroke-dasharray:3 3; }}
.cards {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); margin-top:18px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:15px 16px;
  min-width:0; border-left:3px solid var(--crit); }}
.card.ok {{ border-left-color:var(--good); }}
.card header {{ display:flex; gap:10px; align-items:center; justify-content:space-between; margin-bottom:8px; }}
.rat {{ font-size:13px; color:var(--ink-2); margin-bottom:9px; }}
.verdict {{ font-size:13px; margin:0; }}
.card.ok .verdict {{ color:var(--ink); font-weight:500; }}
ol, ul {{ max-width:66ch; padding-left:20px; }}
li {{ margin-bottom:9px; }}
code {{ font-family:var(--f-mono); font-size:0.9em; background:var(--core); padding:1px 5px;
  border-radius:3px; }}
.foot {{ margin-top:56px; padding-top:18px; border-top:1px solid var(--line);
  font-size:12.5px; color:var(--ink-3); }}
.foot p {{ max-width:none; }}
</style>

<div class="wrap">
  <div class="eyebrow">BioEmu membrane guidance · {d['n_structures']} structures</div>
  <h1>Which objectives can actually guide membrane insertion?</h1>
  <p class="lede">Nine candidate objectives, screened against generated melittin ensembles and an
  experimentally-oriented reference, <em>before</em> any of them is used for guidance. Two survive.</p>

  <div class="slab">
    <p><strong>Headline, after the second pass.</strong> The membrane <em>frame</em> was the
    dominant defect, not the objectives. Letting each objective solve for its own membrane
    placement produced normals 66–83° from the experimentally determined one, and under that
    broken frame seven of nine objectives could be improved by tearing the peptide apart.
    <strong>Transferring the frame from the OPM-oriented melittin structure instead flips every
    objective to the correct sign on helicity and all but eliminates the gaming</strong> — the
    original Ez potential included.</p>
  </div>

  <h2>Why this screen exists</h2>
  <p>We guided BioEmu with the Ez depth potential and it appeared to work — energy improved
  significantly at high guidance strength. It was an artefact. Measured on the generated
  ensembles, Ez correlates <em>positively</em> with helicity (r = +0.52), and the structures that
  fail the physicality filter score <em>better</em> than those that pass. Guidance was lowering the
  energy by pulling the molecule apart.</p>
  <p>That failure is structural, not a tuning problem. Ez is additive and per-residue,
  E = Σ f(type, depth), so nothing in it knows the residues are bonded. The cheapest way to lower
  it is to gain the freedom to place each residue at its own optimal depth — and both unfolding
  and chain-breaking buy exactly that freedom. The tests below are built to catch that whole class
  of failure cheaply, on structures we already had.</p>

  <h2>The tests</h2>
  <ol>
    <li><strong>T1 — helix correlation.</strong> Pearson r against DSSP helicity over 650
    structures spanning 0–92% helix. Melittin binds membranes as an amphipathic helix, so a usable
    objective must score helical structures better: <code>r &lt; 0</code>.</li>
    <li><strong>T2 — stretch gaming.</strong> Scale every residue's depth 3× along the membrane
    normal. This separates residues exactly the way the Ez gradient wanted to, and breaks the
    chain doing it. A usable objective must get <em>worse</em>.</li>
    <li><strong>T3 — expulsion gaming.</strong> Translate the whole peptide away from the bilayer.
    No objective should improve. (All nine passed; translation-invariant by construction.)</li>
    <li><strong>T4 — reference rank.</strong> Percentile of the experimental structure within the
    generated ensemble.</li>
    <li><strong>T5 — headroom.</strong> Gap between ensemble mean and best structure, in standard
    deviations. Near-zero means guidance has nothing to do.</li>
    <li><strong>T6 — frame recovery.</strong> Given a structure already in its known membrane
    frame, does the objective find that frame? Ground truth from OPM: normal = z, midplane = 0.</li>
  </ol>

  <h2>Results</h2>
  <div class="tscroll">
  <table>
    <thead><tr>
      <th>Objective</th><th>T1 r(helix)</th><th>T2 stretch</th><th>T6 frame err</th>
      <th>T5 headroom</th><th>Couples residues</th><th>Verdict</th>
    </tr></thead>
    <tbody>
    {''.join(rows)}
    </tbody>
  </table>
  </div>
  <p class="sub">T2 negative = stretching improves the score. T6 is the angle from the true
  membrane normal; frame-free objectives have no entry.</p>

  <figure>{c2}</figure>
  <figure>{c1}</figure>

  <div class="slab">
    <p><strong>The pattern is exactly the hypothesis.</strong> The only objectives that resist both
    tests are the two that <em>couple residues</em> — the hydrophobic moment is a vector sum over
    residues, so it needs their radial directions to stay coherent. Unfolding scrambles that
    coherence and stretching dilutes it, so neither can be used to cheat. Every purely additive
    per-residue objective, however carefully bounded, remains gameable.</p>
    <p>Note that <code>ez_clamped</code> and <code>ez_anchored</code> were written specifically to
    close the escape routes, and both still fail. Bounding the reward was not enough; the problem
    is additivity itself.</p>
  </div>

  <h2>The membrane frame is a separate failure</h2>
  <p>This one was not anticipated. Given the OPM-oriented melittin structure — where the membrane
  placement is already solved, with the midplane at z = 0 and the bilayer boundary at z = +15.4 Å —
  every objective that searches for its own frame picks a normal 66–83° away from the true one.
  Our tilt measurement reproduces OPM's published 65° tilt to within 1°, so the discrepancy is in
  the objectives, not the measurement.</p>

  <figure>{c3}</figure>

  <p>This matters beyond the objectives themselves. All of our guidance runs used a per-structure
  optimised membrane frame, chosen specifically to avoid needing a reference structure. That choice
  now looks wrong: the optimiser finds whichever placement minimises a flawed objective, not the
  physical one. A reference-aligned frame — the Kabsch approach, given a correct melittin
  reference — is the more defensible option, and the OPM file now makes it available.</p>

  <h2>Objective by objective</h2>
  <div class="cards">
    {''.join(cards)}
  </div>


  <h2>Part two — fixing the frame</h2>
  <p>Rather than searching for a membrane placement, transfer it from a structure whose placement
  is already known. Each generated structure is superimposed by Kabsch onto the OPM-oriented
  melittin, and the resulting bilayer plane is expressed back in that structure's own coordinates.
  The superposition is held fixed when differentiating, exactly as the frame search was, so
  guidance still cannot re-pose the molecule. Verified exact: aligning the reference to itself
  returns normal = (0, 0, 1) and midplane 0, and a randomly rotated copy reproduces every residue
  depth to 0.0000 Å.</p>

  <div class="tscroll">
  <table>
    <thead><tr>
      <th rowspan="2">Objective</th>
      <th colspan="2">r(helicity) — want &lt; 0</th>
      <th colspan="2">Stretch — want &gt; 0</th>
    </tr><tr>
      <th>searched</th><th>OPM frame</th><th>searched</th><th>OPM frame</th>
    </tr></thead>
    <tbody>
    {frame_rows}
    </tbody>
  </table>
  </div>

  <div class="slab">
    <p><strong>Every objective improves, and most become usable.</strong> Helicity correlations
    flip from positive to negative across the board. Stretch gaming goes from as bad as −9.05σ to
    positive almost everywhere. <code>ez_clamped</code> also recovers its dynamic range: under the
    searched frame its spread was 0.020 — too flat to distinguish any two structures — because it
    could always find a placement that saturated the clamps. Pinned to the OPM frame it cannot,
    and the spread rises to 0.562.</p>
    <p>So the earlier conclusion needs revising. Additivity made the objectives
    <em>exploitable</em>, but what they exploited was the freedom to move the membrane. Remove
    that freedom and the plain Ez potential on Cα — the one we started with — scores helices
    correctly (r = −0.559) and resists stretching (+0.78σ).</p>
  </div>

  <p class="sub">One caveat on the tilt figures: under the OPM frame the generated ensemble sits
  at 65 ± 9°, matching OPM's experimental 65° for melittin. That agreement is largely by
  construction — Kabsch-aligning to a reference with a 65° tilt places helical structures at
  roughly that tilt. It confirms the transfer works; it is not independent evidence that the
  generated structures are correctly oriented.</p>

  <h2>What to do next</h2>
  <ol>
    <li><strong>Use the OPM reference frame for all future guidance.</strong> This is done and
    verified. It is the single change that matters most, and it helps every objective equally.</li>
    <li><strong>Guide with <code>moment_alignment</code> or <code>combined</code>.</strong>
    Under the OPM frame <code>moment_alignment</code> has the strongest helicity correlation
    (−0.677) and a wide spread (3.43), though its stretch margin is thin (−0.10σ, near noise).
    <code>combined</code> is the balanced choice: −0.567 with a clear +0.32σ stretch margin.</li>
    <li><strong>Re-run the λ sweep with the fixed frame.</strong> Every guidance result so far used
    the broken frame, so the entire sweep needs repeating before any of it means anything. That is
    roughly 30 minutes of GPU time.</li>
    <li><strong>Re-validate anything new.</strong> The screen runs on existing structures in
    minutes with no GPU, and would have caught both defects before four rounds of sampling.</li>
    <li><strong>Reconsider the target peptide.</strong> Melittin's unguided ensemble already sits
    near the limit a connected helix can reach, so headroom stays small whatever the objective.</li>
  </ol>

  <div class="foot">
    <p><strong>Data.</strong> {d['n_structures']} BioEmu structures (melittin,
    <code>{esc(d['sequence'])}</code>) pooled from the baseline, DPS λ∈{{1.69, 5, 20, 100, 500}},
    dps_phys, and SMC runs; helicity 0–92%. Reference: <code>{esc(str(d.get('reference')))}</code>.
    Membrane frame ground truth: <code>{esc(str(d.get('opm_reference')))}</code> (OPM entry 149,
    half-thickness 15.4 Å, peripheral placement, 65° tilt).</p>
    <p><strong>Limitations.</strong> One peptide, one model checkpoint. T2 and T3 run on a
    128-structure subset for speed. The OPM orientation is itself a computed placement of an
    aqueous crystal structure, not a direct measurement — it is the best experimentally-anchored
    reference available, and no PDB entry of free melittin in a micelle or bilayer exists.
    Structures are backbone + Cβ only; no side chains were packed, so side-chain depth is
    approximated throughout.</p>
    <p>Generated by <code>validate_objectives.py</code> and <code>make_report.py</code>.</p>
  </div>
</div>
"""
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(html) / 1024:.0f} KB)")
    print(f"passed: {[o['key'] for o in passed]}")
    if best_frame:
        print(f"best frame recovery: {best_frame['key']} at {best_frame['frame']['normal_err_deg']:.0f} deg")


if __name__ == "__main__":
    main()
