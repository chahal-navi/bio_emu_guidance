"""Generate opm_frame_sweep_report.html - the follow-up to objective_functions_report.html.

Covers everything after the objective screen: the OPM frame fix, the two bugs it exposed,
the A/B against the searched frame, and the guidance re-run that followed.
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "opm_frame_sweep_report.html"
GOOD, CRIT, WARN = "#0ca30c", "#d03b3b", "#fab219"

SHORT = {
    "ez_ca": "Ez (Ca)", "ez_clamped": "Ez clamped", "ez_anchored": "Ez + anchor",
    "depth_contrast": "Depth contrast", "burial_saturating": "Saturating burial",
    "hydrophobic_moment": "Hydrophobic moment", "moment_alignment": "Moment x normal",
    "combined_moment_heavy": "Burial + 3x moment", "combined": "Burial + moment",
}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def paired_bars(items, *, title, sub, unit, want):
    """items: [(label, before, after)]. Two bars per row, before/after."""
    rowh, gap, left, right, top = 34, 10, 150, 70, 52
    n = len(items)
    h = top + n * (rowh + gap) + 30
    w, = (760,)
    plot_w = w - left - right
    vals = [v for _, a, b in items for v in (a, b) if v is not None]
    lo, hi = min(0, min(vals)), max(0, max(vals))
    span = (hi - lo) or 1
    lo, hi = lo - span * 0.1, hi + span * 0.1
    span = hi - lo

    def X(v):
        return left + (v - lo) / span * plot_w

    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}" style="width:100%;height:auto">',
         f'<text x="0" y="18" class="ct">{esc(title)}</text>',
         f'<text x="0" y="36" class="cs">{esc(sub)}</text>']
    x0 = X(0)
    p.append(f'<line x1="{x0:.1f}" y1="{top - 10}" x2="{x0:.1f}" y2="{h - 26}" class="axis"/>')
    for i, (label, before, after) in enumerate(items):
        y = top + i * (rowh + gap)
        p.append(f'<text x="{left - 12}" y="{y + 20}" class="cl" text-anchor="end">{esc(label)}</text>')
        for j, (v, op) in enumerate(((before, 0.38), (after, 1.0))):
            if v is None:
                continue
            xv = X(v)
            bx, bw = min(x0, xv), abs(xv - x0)
            good = (v < 0) if want == "neg" else (v > 0)
            p.append(f'<rect x="{bx:.1f}" y="{y + 2 + j * 15}" width="{max(bw, 1.5):.1f}" '
                     f'height="12" rx="2" fill="{GOOD if good else CRIT}" opacity="{op}"/>')
            anc = "start" if v >= 0 else "end"
            p.append(f'<text x="{xv + (6 if v >= 0 else -6):.1f}" y="{y + 12 + j * 15}" '
                     f'class="cv" text-anchor="{anc}">{v:+.2f}{unit}</text>')
    p.append(f'<text x="{left}" y="{h - 10}" class="cl">pale = searched frame · solid = OPM frame</text>')
    p.append("</svg>")
    return "\n".join(p)


def main():
    fc = json.loads((HERE / "objective_validation" / "frame_comparison.json").read_text())
    sweep = list(csv.DictReader((HERE / "results_opm" / "analysis" / "summary.csv").open()))

    order = ["ez_ca", "ez_clamped", "ez_anchored", "depth_contrast", "burial_saturating",
             "hydrophobic_moment", "moment_alignment", "combined_moment_heavy", "combined"]
    by_key = {o["key"]: o for o in fc["objectives"]}
    rows_r = [(SHORT[k], by_key[k]["r_opt"], by_key[k]["r_opm"]) for k in order if k in by_key]
    rows_s = [(SHORT[k], by_key[k].get("stretch_opt"), by_key[k].get("stretch_opm"))
              for k in order if k in by_key and by_key[k].get("stretch_opt") is not None]

    c1 = paired_bars(rows_r, title="Helicity correlation, before and after the frame fix",
                     sub="Pearson r against DSSP helicity. Negative means the objective prefers folded helices, which is what melittin's membrane-bound state is.",
                     unit="", want="neg")
    c2 = paired_bars(rows_s, title="Stretch margin, before and after the frame fix",
                     sub="Score change when the chain is pulled 3x along the membrane normal, in ensemble sigma. Positive means stretching makes the score worse, i.e. it cannot be used to cheat.",
                     unit="σ", want="pos")

    def srow(r):
        phys, cn = float(r["physical_frac"]), float(r["break_cn_frac"])
        col = GOOD if phys > 0.8 else (WARN if phys > 0.5 else CRIT)
        name = r["method"].replace("dps_", "").replace("_lam", "  λ=")
        return (f"<tr><td>{esc(name)}</td>"
                f"<td class='num' style='color:{col}'>{phys:.2f}</td>"
                f"<td class='num'>{cn:.2f}</td>"
                f"<td class='num'>{float(r['clash_frac']):.2f}</td>"
                f"<td class='num'>{float(r['ez_kcal_mean']):.2f}</td>"
                f"<td class='num'>{float(r['helicity_mean']):.2f}</td></tr>")

    sweep_rows = "".join(srow(r) for r in sorted(sweep, key=lambda r: r["method"]))

    html = f"""<meta name="claude:viewport-fit-sides" content="cover">
<title>OPM Frame and Guidance Rerun</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: single-column follow-up report, matching objective_functions_report.html */
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
h3 {{ font-family:var(--f-disp); font-size:15.5px; margin:26px 0 4px; }}
p {{ margin:0 0 14px; max-width:66ch; }}
.eyebrow {{ font-size:11px; letter-spacing:0.14em; text-transform:uppercase;
  color:var(--accent); font-weight:600; margin-bottom:14px; }}
.lede {{ font-size:17px; color:var(--ink-2); max-width:64ch; }}
.sub {{ color:var(--ink-3); font-size:13px; margin:-2px 0 18px; }}
.slab {{ background:var(--core); border-top:2px solid var(--accent); border-bottom:2px solid var(--accent);
  padding:20px 22px; margin:26px 0; border-radius:2px; }}
.slab p:last-child {{ margin-bottom:0; }}
.slab.bad {{ border-color:var(--crit); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
.tscroll {{ overflow-x:auto; margin:18px 0 8px; }}
th {{ text-align:left; font-family:var(--f-disp); font-size:11.5px; letter-spacing:0.05em;
  text-transform:uppercase; color:var(--ink-3); font-weight:600;
  border-bottom:1px solid var(--line); padding:0 12px 8px 0; white-space:nowrap; }}
td {{ border-bottom:1px solid var(--line); padding:10px 12px 10px 0; vertical-align:top; }}
.num {{ font-family:var(--f-mono); font-variant-numeric:tabular-nums; white-space:nowrap; }}
figure {{ margin:22px 0 10px; background:var(--panel); border:1px solid var(--line);
  border-radius:6px; padding:18px; }}
.ct {{ font-family:var(--f-disp); font-size:14px; font-weight:700; fill:var(--ink); }}
.cs {{ font-size:11px; fill:var(--ink-3); }}
.cl {{ font-size:11.5px; fill:var(--ink-2); }}
.cv {{ font-size:10.5px; fill:var(--ink-2); font-family:var(--f-mono); }}
.axis {{ stroke:var(--ink-3); stroke-width:1.2; }}
.diagram {{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:16px; margin:20px 0; overflow-x:auto; }}
ol, ul {{ max-width:66ch; padding-left:20px; }}
li {{ margin-bottom:10px; }}
code {{ font-family:var(--f-mono); font-size:0.9em; background:var(--core); padding:1px 5px;
  border-radius:3px; }}
.foot {{ margin-top:56px; padding-top:18px; border-top:1px solid var(--line);
  font-size:12.5px; color:var(--ink-3); }}
.foot p {{ max-width:none; }}
</style>

<div class="wrap">
  <div class="eyebrow">BioEmu membrane guidance · follow-up to the objective screen</div>
  <h1>The frame fix worked. The guidance still failed.</h1>
  <p class="lede">Replacing the searched membrane frame with one transferred from an
  experimentally-oriented structure repaired every objective on paper — then the guidance re-run
  destroyed 74–98% of structures anyway. This explains the gap.</p>

  <div class="slab bad">
    <p><strong>In short.</strong> Transferring the membrane frame from the OPM melittin structure
    flipped all nine objectives to the correct sign on helicity and removed the stretch exploit.
    Guiding with the two best of them nonetheless collapsed physical validity from 0.94 to
    <strong>0.02</strong>. The screen validated the objectives on BioEmu's <em>finished</em>
    structures, but guidance acts on noisy intermediates — and the failure that actually occurred
    is one no objective can fix, because it comes from guiding positions while leaving the
    rotational frames untouched.</p>
  </div>

  <h2>What the OPM reference actually is</h2>
  <p>OPM (Orientations of Proteins in Membranes) takes a deposited structure and computes where it
  sits in an implicit bilayer, then republishes the coordinates <strong>already rotated and
  translated into that membrane frame</strong>. For <code>2mlt_opm.pdb</code> the file states its
  geometry directly in the header:</p>
  <div class="diagram"><pre style="margin:0;font-family:var(--f-mono);font-size:12px">REMARK      1/2 of bilayer thickness:   15.4</pre></div>
  <p>and carries 277 <code>DUM</code> pseudo-atoms, all at <strong>z = +15.400</strong>, marking
  the hydrocarbon boundary. The midplane is the z = 0 plane. Melittin's own Cα atoms span
  z = 1.76 to 20.09 Å, so it lies across the interface at a 65° tilt — a peripheral peptide
  resting on one leaflet, not a transmembrane one, which is why only a single boundary plane
  is drawn.</p>

  <h3>Is the membrane fixed?</h3>
  <p>Fixed <em>relative to that one structure</em>. The OPM file's membrane is the z = 0 plane in
  the OPM file's own coordinate system, and that is useless on its own, because BioEmu emits
  structures in arbitrary poses — the model is SE(3)-equivariant, so a generated structure has no
  canonical orientation to compare against.</p>
  <p>So the membrane is <strong>transferred</strong>. Each generated structure is superimposed onto
  the OPM reference by Kabsch, which says how that structure would have to be rotated to match the
  reference's pose. We then carry the membrane plane <em>back</em> through that transformation and
  express it in the generated structure's own coordinates, as a normal <strong>n</strong> and an
  offset <strong>c</strong>, giving each residue a depth <code>z = n·x − c</code>. Mathematically
  this is identical to moving the structure into the OPM frame, but it leaves the coordinates
  untouched, which keeps the gradient clean. Verified exact: self-alignment returns n = (0, 0, 1)
  and c = 0, and a randomly rotated copy reproduces every residue depth to 0.0000 Å.</p>

  <h3>How that differs from searching</h3>
  <div class="tscroll">
  <table>
    <thead><tr><th></th><th>Searched frame</th><th>OPM reference frame</th></tr></thead>
    <tbody>
      <tr><td>Where the membrane comes from</td>
        <td>chosen to minimise the objective</td>
        <td>transferred from an experimentally-anchored structure</td></tr>
      <tr><td>Can the objective move it?</td>
        <td><strong style="color:var(--crit)">Yes — this was the defect</strong></td>
        <td><strong style="color:var(--good)">No</strong></td></tr>
      <tr><td>Needs a reference?</td><td>no</td><td>yes, of the same peptide</td></tr>
      <tr><td>Recovers the true normal?</td>
        <td>66–83° off</td><td>exact, by construction</td></tr>
    </tbody>
  </table>
  </div>
  <p>The searched frame gave the objective an escape hatch: for any conformation, however bad, it
  could find the membrane placement that scored best. "Improve the score" was therefore satisfiable
  by moving the membrane instead of improving the structure. A transferred frame is determined by
  shape-matching to the reference, which the objective has no influence over, so the only remaining
  way to improve the score is to change the conformation. That is why every correlation flipped.</p>

  <h2>The A/B, on 650 existing structures</h2>
  <figure>{c1}</figure>
  <figure>{c2}</figure>
  <p>Every objective improves. Helicity correlations go negative across the board — including
  <code>ez_ca</code>, the unmodified Ez potential we started with, at −0.559. The worst stretch
  exploit, <code>depth_contrast</code> at −9.05σ, becomes +0.46σ. <code>ez_clamped</code> also
  recovers its dynamic range (0.020 → 0.562): under the searched frame it could always find a
  placement that saturated its clamps.</p>

  <h3>Two bugs this exposed</h3>
  <ol>
    <li><strong><code>ez_ca</code> silently ignored externally supplied frames.</strong> It
    delegated to <code>EzMembranePotential.evaluate()</code>, which always solves its own. Its
    first "OPM frame" result came back byte-identical to the searched one, which is what gave it
    away.</li>
    <li><strong><code>moment_alignment</code> had its sign inverted.</strong> With the OPM normal
    pointing away from the bilayer core, a hydrophobic face turned <em>toward</em> the core must
    give μ·n &lt; 0. It had been rewarding the face pointing at solvent.</li>
  </ol>

  <h2>The guidance re-run</h2>
  <p>Seven runs, 50 structures each: baseline, plus <code>combined</code> and
  <code>moment_alignment</code> at λ ∈ {{20, 100, 500}}, all under the OPM frame. The physicality
  potentials were deliberately left off — the screen predicted both objectives resist stretching,
  and adding chain restraints would have masked whether that prediction held.</p>

  <div class="tscroll">
  <table>
    <thead><tr><th>Run</th><th>Physical</th><th>CN break</th><th>Clash</th><th>Ez</th><th>Helicity</th></tr></thead>
    <tbody>{sweep_rows}</tbody>
  </table>
  </div>
  <p class="sub">At λ ≥ 100 only one structure of fifty survives the physicality filter.</p>

  <div class="slab bad">
    <p><strong>It failed at every λ.</strong> Physical validity falls from 0.94 to 0.24–0.26 at
    λ = 20 and to 0.02 at λ = 100 and above. <code>combined</code> did not improve its energy at
    all (−2.69 against the baseline's −2.68). <code>moment_alignment</code> reached −3.26, but with
    helicity <strong>0.00</strong> and one surviving structure — the original pathology exactly:
    energy down, molecule destroyed.</p>
    <p>λ = 100 and λ = 500 differ by 0.001 nm in final coordinates, so the guidance clip saturates:
    above roughly λ = 100 the strength knob does nothing and only the direction matters.</p>
  </div>

  <h2>Why the screen did not predict this</h2>
  <p>The screen asked whether stretching a <em>realistic, finished</em> BioEmu structure improves
  the score. It does not — that result stands. But guidance never touches finished structures. It
  acts on the Tweedie estimate x̂₀ at intermediate noise levels, where x̂₀ is not a realistic
  peptide at all. An objective's behaviour on the final distribution says nothing about its
  behaviour on noisy intermediates. The screen tested the wrong distribution.</p>

  <h2>What the failure signature says</h2>
  <p>Look at which test fails: <strong>CN breaks at 0.74–0.98, clashes only 0.04–0.10.</strong>
  Peptide bonds, not steric overlap. That points somewhere specific.</p>
  <p>BioEmu stores each residue as a rigid body — a Cα position plus a rotation matrix — and
  reconstructs N, C, O and Cβ from that pair. Nothing enforces that C(i) and N(i+1) land 1.33 Å
  apart; the network has merely <em>learned</em> to emit consecutive frames whose reconstructed
  atoms form proper bonds. All of our guidance pushes <code>pos</code> and never touches
  <code>node_orientations</code>, so the two channels drift out of agreement, and the reconstructed
  peptide bond is exactly where that disagreement becomes visible. No choice of objective repairs
  this, because it is not about the objective. It is about which channel is being guided.</p>

  <h2>What follows</h2>
  <ol>
    <li><strong>Guide the rotational channel.</strong> Rotating a residue's frame moves Cβ without
    moving any Cα, so it <em>structurally cannot</em> stretch a bond. This is the one approach
    immune to the failure that has now killed every run.
    <code>rot_guidance_unverified.py</code> is the starting point, and
    <code>moment_alignment</code> is already orientation-dependent, which such guidance requires.</li>
    <li><strong>Re-validate on noised intermediates.</strong> The screen is salvageable: add x̂₀
    estimates at several noise levels to its test set. That is a cheap repair to a methodology
    error, and it would have predicted this failure.</li>
    <li><strong>Put the physicality terms back on.</strong> They measurably helped before (maximum
    stretch 16.3 Å → 6.4 Å), and there is no longer any reason to run without them.</li>
  </ol>

  <div class="foot">
    <p><strong>Data.</strong> A/B on {fc['n_structures']} pooled structures from the earlier sweeps.
    Guidance re-run: 7 × 50 structures, melittin, BioEmu v1.1, RTX 4090, shared seed.
    Frame reference <code>{esc(fc['opm_reference'])}</code> (OPM entry 149, half-thickness 15.4 Å,
    peripheral, 65° tilt). Raw data in <code>results_opm/</code>.</p>
    <p><strong>Caveat on tilt.</strong> Under the OPM frame the ensemble sits at 65 ± 9°, matching
    OPM's experimental 65°. That agreement is largely by construction — Kabsch-aligning to a
    65°-tilt reference places helical structures at about that tilt. It confirms the transfer
    works; it is not independent evidence that generated structures are correctly oriented.</p>
    <p>Generated by <code>compare_frames.py</code>, <code>run.py</code>, <code>analyze.py</code>
    and <code>make_sweep_report.py</code>.</p>
  </div>
</div>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
