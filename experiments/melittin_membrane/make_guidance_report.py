"""Generate 03_guidance_results_report.html - what guidance actually achieved.

Third report. Covers the configuration that finally worked (Ez on Ca under the OPM
reference frame), the statistical error that had hidden it, and the lambda curve.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
OUT = REPORTS / "03_guidance_results_report.html"
GOOD, CRIT, WARN = "#0ca30c", "#d03b3b", "#fab219"

# lambda, paired effect, physical fraction, pairs kept of 50, p
CURVE = [
    (20, -0.174, 0.94, 44, "4e-08"),
    (35, -0.191, 0.78, 37, "1e-07"),
    (50, -0.213, 0.70, 33, "3e-07"),
    (70, -0.250, 0.68, 32, "1e-07"),
    (100, -0.298, 0.54, 26, "2e-06"),
    (500, -0.415, 0.40, 19, "9e-07"),
]


def curve_chart():
    """Effect against retained fraction, with lambda labelled on each point."""
    w, h = 720, 330
    l, r, t, b = 64, 150, 44, 50
    pw, ph = w - l - r, h - t - b
    xs = [p[3] / 50 for p in CURVE]          # yield
    ys = [abs(p[1]) for p in CURVE]          # effect magnitude
    x0, x1 = 0.3, 1.0
    y0, y1 = 0.0, 0.46

    def X(v):
        return l + (v - x0) / (x1 - x0) * pw

    def Y(v):
        return t + ph - (v - y0) / (y1 - y0) * ph

    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="effect versus yield" '
         f'style="width:100%;height:auto">',
         f'<text x="0" y="16" class="ct">Effect against ensemble yield</text>',
         f'<text x="0" y="33" class="cs">Each point is one guidance strength. Up = bigger effect per structure; right = more structures survive.</text>']
    # grid + axes
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4]:
        y = Y(frac)
        p.append(f'<line x1="{l}" y1="{y:.1f}" x2="{l + pw}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{l - 10}" y="{y + 4:.1f}" class="cl" text-anchor="end">{frac:.1f}</text>')
    for frac in [0.4, 0.6, 0.8, 1.0]:
        x = X(frac)
        p.append(f'<text x="{x:.1f}" y="{h - 26}" class="cl" text-anchor="middle">{frac:.0%}</text>')
    p.append(f'<text x="{l + pw / 2:.0f}" y="{h - 8}" class="cl" text-anchor="middle">structures surviving the physicality filter</text>')
    p.append(f'<text x="14" y="{t + ph / 2:.0f}" class="cl" text-anchor="middle" '
             f'transform="rotate(-90 14 {t + ph / 2:.0f})">|effect| (kcal/mol)</text>')
    # constant effect-x-yield contour: the trade-off the data actually follows
    pts = []
    for i in range(40):
        xv = 0.34 + i * (0.99 - 0.34) / 39
        pts.append(f"{X(xv):.1f},{Y(0.152 / xv):.1f}")
    p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="var(--ink-3)" '
             f'stroke-width="1.3" stroke-dasharray="5 4"/>')
    p.append(f'<text x="{X(0.42):.1f}" y="{Y(0.152 / 0.42) - 10:.1f}" class="cl" '
             f'style="fill:var(--ink-3)">constant effect x yield</text>')
    # the measured points
    for (lam, eff, phys, pairs, _), xv, yv in zip(CURVE, xs, ys):
        cx, cy = X(xv), Y(yv)
        col = GOOD if pairs >= 44 else (WARN if pairs >= 32 else CRIT)
        p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{col}" '
                 f'stroke="var(--panel)" stroke-width="2"/>')
        p.append(f'<text x="{cx + 11:.1f}" y="{cy + 4:.1f}" class="cv">λ={lam}</text>')
    p.append("</svg>")
    return "\n".join(p)


def main():
    REPORTS.mkdir(exist_ok=True)
    rows = ""
    for lam, eff, phys, pairs, pv in CURVE:
        col = GOOD if pairs >= 44 else (WARN if pairs >= 32 else CRIT)
        rows += (f"<tr><td class='num'>{lam}</td>"
                 f"<td class='num'>{eff:+.3f}</td>"
                 f"<td class='num' style='color:{col}'>{phys:.2f}</td>"
                 f"<td class='num'>{pairs}/50</td>"
                 f"<td class='num'>100%</td>"
                 f"<td class='num'>{pv}</td>"
                 f"<td class='num'>{abs(eff) * pairs / 50:.3f}</td></tr>")

    html = f"""<meta name="claude:viewport-fit-sides" content="cover">
<title>Guidance Results</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: single-column results report, third in the series */
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
.slab.win {{ border-color:var(--good); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
.tscroll {{ overflow-x:auto; margin:18px 0 8px; }}
th {{ text-align:left; font-family:var(--f-disp); font-size:11.5px; letter-spacing:0.05em;
  text-transform:uppercase; color:var(--ink-3); font-weight:600;
  border-bottom:1px solid var(--line); padding:0 14px 8px 0; white-space:nowrap; }}
td {{ border-bottom:1px solid var(--line); padding:10px 14px 10px 0; }}
.num {{ font-family:var(--f-mono); font-variant-numeric:tabular-nums; white-space:nowrap; }}
figure {{ margin:22px 0 10px; background:var(--panel); border:1px solid var(--line);
  border-radius:6px; padding:18px; }}
.ct {{ font-family:var(--f-disp); font-size:14px; font-weight:700; fill:var(--ink); }}
.cs {{ font-size:11px; fill:var(--ink-3); }}
.cl {{ font-size:11.5px; fill:var(--ink-2); }}
.cv {{ font-size:11px; fill:var(--ink-2); font-family:var(--f-mono); }}
.grid {{ stroke:var(--line); stroke-width:1; }}
.calc {{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:14px 16px; margin:16px 0; font-family:var(--f-mono); font-size:12.5px;
  overflow-x:auto; line-height:1.75; }}
.calc .hi {{ color:var(--accent); font-weight:500; }}
ol, ul {{ max-width:66ch; padding-left:20px; }}
li {{ margin-bottom:10px; }}
code {{ font-family:var(--f-mono); font-size:0.9em; background:var(--core); padding:1px 5px;
  border-radius:3px; }}
.foot {{ margin-top:56px; padding-top:18px; border-top:1px solid var(--line);
  font-size:12.5px; color:var(--ink-3); }}
.foot p {{ max-width:none; }}
</style>

<div class="wrap">
  <div class="eyebrow">BioEmu membrane guidance · report 3 of 3</div>
  <h1>Guidance works, consistently, and hits a ceiling</h1>
  <p class="lede">The Ez potential under the OPM reference frame improves every structure it
  touches, at every strength tested. The effect is real, highly significant, dose-dependent —
  and bounded by a fixed exchange rate between improvement and structural damage.</p>

  <div class="slab win">
    <p><strong>Headline.</strong> At λ = 20, guidance improves the membrane energy of
    <strong>44 out of 44</strong> structures (paired p = 4×10⁻⁸) at <strong>zero</strong> structural
    cost — physical validity 0.94, identical to baseline. Raising λ buys a larger per-structure
    effect and costs proportionally more broken structures, at a rate so constant that the total
    useful improvement per 50 samples never changes. That looks like a ceiling on position-only
    guidance, not a tuning problem.</p>
  </div>

  <h2>What finally worked</h2>
  <p>The configuration is the unmodified Senes Ez potential evaluated on Cα atoms, with the
  membrane frame transferred by Kabsch superposition from the OPM-oriented melittin structure
  rather than searched for. Both halves matter, and neither was obvious:</p>
  <ul>
    <li>Report 1 found that letting the objective choose its own membrane placement gave it an
    escape hatch — it could improve its score by moving the membrane rather than fixing the
    structure.</li>
    <li>Report 2 fixed that, and found that <code>ez_ca</code> — which had looked mediocre under
    the broken frame — was in fact the best objective available, passing all three screens.</li>
  </ul>
  <p>Ez under the OPM frame improves 100% of structures at every λ. Under the searched frame the
  same objective improves only 60–68% of them, which is what an exploitable objective looks like
  once the metric no longer shares the exploit.</p>

  <h2>The statistical error that hid it</h2>
  <p>Every significance test before this point was wrong, and the error ran in both directions.
  Runs in a sweep share <code>base_seed</code>, and <code>generate_batch</code> seeds each batch
  as <code>base_seed + start_idx</code>, so sample <em>j</em> of the guided run and sample
  <em>j</em> of the baseline start from the <strong>same prior draw</strong>. They are matched
  pairs. Testing them as independent samples throws away nearly all the signal.</p>

  <div class="calc">
same 44 pairs, same numerator (<span class="hi">-0.1744</span>), two denominators:<br><br>
UNPAIRED   SE = sqrt(s_g²/n + s_b²/n) = sqrt(0.9426/44 + 1.2137/44) = 0.2214<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t  = -0.1744 / 0.2214 = -0.788&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p = <span class="hi">0.43</span><br><br>
PAIRED&nbsp;&nbsp;&nbsp;&nbsp; SE = sd(d)/sqrt(n)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = 0.1742/sqrt(44)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = 0.0263<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t  = -0.1744 / 0.0263 = -6.641&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; p = <span class="hi">4.3e-08</span><br><br>
why sd(d) collapses:<br>
&nbsp;&nbsp;Var(g-b) = Var(g) + Var(b) - 2·rho·s_g·s_b<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; = 0.9426 + 1.2137 - 2(<span class="hi">0.9938</span>)(0.9709)(1.1017) = 0.0303<br>
&nbsp;&nbsp;the variances sum to 2.16; the covariance term removes 2.13 of it.
  </div>

  <p>ρ = 0.994 because guidance <em>nudges</em> each trajectory rather than producing an unrelated
  structure. It is the same move as a control variate, or evaluating two models on identical
  minibatches. The unpaired test asked "is a random guided structure better than a random baseline
  one?" — unanswerable at n = 44 when between-structure spread is 1.1 and the effect is 0.17. The
  paired test asks "when I guide <em>this</em> trajectory, does it improve?" Answer: every time.</p>
  <p class="sub">This is only legitimate because the seeds match. A run with a different
  <code>base_seed</code> has no real pairing, and a paired test on it would be invalid.</p>

  <h2>The λ curve</h2>
  <div class="tscroll">
  <table>
    <thead><tr><th>λ</th><th>Effect (paired)</th><th>Physical</th><th>Pairs kept</th>
      <th>Won</th><th>p</th><th>Effect × yield</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
  <p class="sub">Effect in kcal/mol, scored under the OPM frame on pairs where both structures
  pass the physicality filter. "Won" is the share of pairs that improved.</p>

  <figure>{curve_chart()}</figure>

  <div class="slab">
    <p><strong>There is no sweet spot — the exchange rate is fixed.</strong> Effect × yield, the
    total improvement surviving per 50 samples generated, stays between 0.141 and 0.160 across a
    25× range of λ. The measured points sit on a constant-product contour. Turning λ up buys a
    larger per-structure effect and costs proportionally more structures, and the product does not
    move.</p>
    <p>The one departure is at the very start. λ = 20 returns 0.029 of effect per structure lost,
    roughly twice the efficiency of anything above it; going from λ = 20 to λ = 35 costs seven more
    broken structures and buys only 0.017 more effect. <strong>λ = 20 is the operating
    point</strong>: nearly all the accessible effect, at the only structural cost near zero.</p>
  </div>

  <h2>Why this reads as a ceiling</h2>
  <p>A constant exchange rate means every additional unit of guidance force splits the same way
  between improving the conformation and pulling the residue frames out of agreement. That is
  exactly what you would expect if the damage mechanism is intrinsic to the channel being guided.</p>
  <p>BioEmu stores each residue as a Cα position plus a rotation, and reconstructs the backbone
  from both. Nothing enforces that C(i) and N(i+1) land 1.33 Å apart — the network has only
  <em>learned</em> to emit consistent consecutive frames. All of our guidance moves
  <code>pos</code> and never touches <code>node_orientations</code>, so the two channels drift
  apart and the reconstructed peptide bond is where it shows. The failure signature agrees:
  CN breaks dominate at every λ while clashes stay flat near baseline.</p>
  <p>Keep the magnitude in perspective too. −0.174 kcal/mol is about <strong>0.3 kT</strong> at
  room temperature: real, perfectly consistent, highly significant, and small.</p>

  <h2>What follows</h2>
  <ol>
    <li><strong>Guide the rotational channel.</strong> Rotating a residue's frame moves Cβ without
    moving any Cα, so it cannot stretch a bond and should not face this exchange rate at all. The
    λ curve makes that testable: if rotational guidance shows a different effect-versus-damage
    slope, the position channel was the binding constraint.</li>
    <li><strong>Test peptides with more headroom.</strong> Melittin's unguided ensemble already
    sits near the limit a connected helix can reach. A transmembrane peptide, whose aqueous
    ensemble should differ sharply from its membrane state, gives guidance more to do.</li>
    <li><strong>Re-validate the screen on noised intermediates.</strong> Every screen so far
    evaluated finished structures, but guidance acts on x̂₀ at intermediate noise. That gap is why
    the screen passed objectives that then failed in practice.</li>
  </ol>

  <div class="foot">
    <p><strong>Data.</strong> Melittin (<code>GIGAVLKVLTTGLPALISWIKRKRQQ</code>), BioEmu v1.1,
    RTX 4090, 50 samples per arm, shared seed. λ ∈ {{20, 35, 50, 70, 100, 500}} across
    <code>results_ez_opm/</code>, <code>results_lam50/</code> and <code>results_lam/</code>.
    Frame reference <code>2mlt_opm.pdb</code> (OPM entry 149, half-thickness 15.4 Å).
    Statistics from <code>paired_analysis.py</code>.</p>
    <p><strong>Limitations.</strong> One peptide, one checkpoint, n = 50 per arm. Results at
    λ ≥ 100 are not interpretable in either direction: filtered they are survivorship-biased
    (48–62% of pairs discarded), unfiltered they are contaminated by broken structures whose
    energies are meaningless. Only λ ≤ 35, where few pairs drop, supports a clean reading.</p>
  </div>
</div>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
