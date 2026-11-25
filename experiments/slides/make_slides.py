"""Build the Phase 1 thesis slide deck.

    python experiments/slides/make_slides.py

Writes phase1_slides.html (gitignored). Charts are drawn to scale from the measured
numbers rather than sketched, so the deck cannot drift from the results.
"""

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "phase1_slides.html"

BLUE, ORANGE, AQUA, YELLOW, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0ca30c", "#d03b3b"

# --- Ez parameters (Senes et al. 2007, Table 1) ------------------------------------
EZ = {"LEU": (-0.64, 17.34, 8.61), "LYS": (1.66, 11.11, 2.09)}
TRP = (-0.85, 11.65, 7.20)

# --- measured results --------------------------------------------------------------
LAMBDA_CURVE = [(20, 0.174, 44), (35, 0.191, 37), (50, 0.213, 33),
                (70, 0.250, 32), (100, 0.298, 26), (500, 0.415, 19)]
PEPTIDES = [("pardaxin\nsurface", 0.266, 0.086, 48), ("melittin", 0.174, 0.022, 44),
            ("pardaxin\nTM", 0.173, 0.025, 40), ("magainin-2", 0.098, 0.086, 43)]
PHYS_COLLAPSE = [("baseline", 0.94), ("λ=20", 0.98), ("λ=100", 0.72), ("λ=500", 0.38)]


def svg(w, h, body, label=""):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}" '
            f'style="width:100%;height:auto;max-width:100%">{body}</svg>')


def chart_diffusion():
    """Noise -> structure, with the two channels BioEmu diffuses."""
    p = []
    for i, (lbl, noise) in enumerate([("pure noise", 1.0), ("", 0.6), ("", 0.3), ("structure", 0.02)]):
        cx = 110 + i * 200
        import random
        random.seed(i + 7)
        pts = []
        for k in range(14):
            t = k / 13
            # a helix that emerges as noise falls
            hx = cx - 42 + t * 84
            hy = 118 + 26 * math.sin(t * 7.5)
            pts.append((hx + random.uniform(-1, 1) * 46 * noise,
                        hy + random.uniform(-1, 1) * 46 * noise))
        d = " ".join(f"{'M' if j == 0 else 'L'}{x:.0f},{y:.0f}" for j, (x, y) in enumerate(pts))
        col = BLUE if i == 3 else "var(--ink-3)"
        p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{3 if i==3 else 2}" '
                 f'stroke-linecap="round" opacity="{0.45 + 0.55*(1-noise):.2f}"/>')
        for x, y in pts:
            p.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3.4" fill="{col}" opacity="{0.45+0.55*(1-noise):.2f}"/>')
        if lbl:
            p.append(f'<text x="{cx}" y="208" class="cap" text-anchor="middle">{lbl}</text>')
        if i < 3:
            p.append(f'<path d="M{cx+92},150 L{cx+104},150" stroke="var(--ink-3)" stroke-width="2" '
                     f'marker-end="url(#ar)"/>')
    p.append('<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
             '<path d="M0,0 L0,6 L8,3 z" fill="var(--ink-3)"/></marker></defs>')
    p.append('<text x="430" y="240" class="cap" text-anchor="middle" style="fill:var(--accent)">'
             'reverse process: remove a little noise, 100 times</text>')
    return svg(860, 252, "".join(p), "diffusion from noise to structure")


def chart_ez():
    """Ez energy against depth for a hydrophobic and a charged residue."""
    w, h, l, r, t, b = 820, 330, 76, 180, 30, 56
    pw, ph = w - l - r, h - t - b
    zmax, ymin, ymax = 34, -1.0, 1.8

    def X(z):
        return l + z / zmax * pw

    def Y(e):
        return t + ph - (e - ymin) / (ymax - ymin) * ph
    p = [f'<rect x="{l}" y="{t}" width="{X(15.4)-l:.0f}" height="{ph}" fill="var(--accent)" opacity="0.07"/>',
         f'<line x1="{X(15.4):.0f}" y1="{t}" x2="{X(15.4):.0f}" y2="{t+ph}" stroke="var(--accent)" '
         f'stroke-width="1.4" stroke-dasharray="5 4"/>',
         f'<text x="{X(15.4)+8:.0f}" y="{t+16}" class="cap" style="fill:var(--accent)">membrane edge</text>',
         f'<text x="{X(5):.0f}" y="{t+ph-8}" class="cap" style="fill:var(--accent)">inside the membrane</text>',
         f'<line x1="{l}" y1="{Y(0):.1f}" x2="{l+pw}" y2="{Y(0):.1f}" stroke="var(--line)" stroke-width="1.5"/>']
    for name, (dE0, zmid, n), col, lab in [("LEU", EZ["LEU"], AQUA, "Leucine — oily"),
                                           ("LYS", EZ["LYS"], ORANGE, "Lysine — charged")]:
        pts = [f"{X(z):.1f},{Y(dE0 / (1 + (max(z,0.01)/zmid)**n)):.1f}" for z in [i*0.4 for i in range(86)]]
        p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="3"/>')
        ey = dE0 / (1 + (0.4/zmid)**n)
        p.append(f'<text x="{l+pw+12}" y="{Y(ey)+5:.1f}" class="cap" style="fill:{col}">{lab}</text>')
    for e in [-1, 0, 1]:
        p.append(f'<text x="{l-12}" y="{Y(e)+4:.0f}" class="cap" text-anchor="end">{e:+d}</text>')
    for z in [0, 10, 20, 30]:
        p.append(f'<text x="{X(z):.0f}" y="{h-30}" class="cap" text-anchor="middle">{z}</text>')
    p.append(f'<text x="{l+pw/2:.0f}" y="{h-8}" class="cap" text-anchor="middle">distance from the middle of the membrane (Å)</text>')
    p.append(f'<text x="18" y="{t+ph/2:.0f}" class="cap" text-anchor="middle" transform="rotate(-90 18 {t+ph/2:.0f})">energy (lower = happier)</text>')
    return svg(w, h, "".join(p), "Ez energy against depth")


def chart_lambda():
    w, h, l, r, t, b = 820, 340, 78, 60, 26, 58
    pw, ph = w - l - r, h - t - b
    x0, x1, y0, y1 = 0.3, 1.0, 0.0, 0.46

    def X(v):
        return l + (v - x0) / (x1 - x0) * pw

    def Y(v):
        return t + ph - (v - y0) / (y1 - y0) * ph
    p = []
    for g in [0.1, 0.2, 0.3, 0.4]:
        p.append(f'<line x1="{l}" y1="{Y(g):.0f}" x2="{l+pw}" y2="{Y(g):.0f}" stroke="var(--line)"/>')
        p.append(f'<text x="{l-12}" y="{Y(g)+4:.0f}" class="cap" text-anchor="end">{g:.1f}</text>')
    for f in [0.4, 0.6, 0.8, 1.0]:
        p.append(f'<text x="{X(f):.0f}" y="{h-32}" class="cap" text-anchor="middle">{f:.0%}</text>')
    pts = [f"{X(x):.1f},{Y(0.152/x):.1f}" for x in [0.34 + i*0.017 for i in range(39)]]
    p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="var(--ink-3)" stroke-width="1.6" stroke-dasharray="6 5"/>')
    p.append(f'<text x="{X(0.45):.0f}" y="{Y(0.152/0.45)-14:.0f}" class="cap" style="fill:var(--ink-3)">same total payoff everywhere on this line</text>')
    for lam, eff, pairs in LAMBDA_CURVE:
        cx, cy = X(pairs/50), Y(eff)
        col = GREEN if pairs >= 44 else (YELLOW if pairs >= 32 else RED)
        p.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="8" fill="{col}" stroke="var(--bg)" stroke-width="2.5"/>')
        p.append(f'<text x="{cx+14:.0f}" y="{cy+5:.0f}" class="cap">λ={lam}</text>')
    p.append(f'<text x="{l+pw/2:.0f}" y="{h-8}" class="cap" text-anchor="middle">structures that survive intact →</text>')
    p.append(f'<text x="18" y="{t+ph/2:.0f}" class="cap" text-anchor="middle" transform="rotate(-90 18 {t+ph/2:.0f})">↑ effect per structure</text>')
    return svg(w, h, "".join(p), "effect against yield")


def chart_peptides():
    w, h, l, r, t = 820, 300, 150, 150, 36
    rowh, gap = 46, 14
    pw = w - l - r
    mx = 0.30

    def X(v):
        return l + v / mx * pw
    p = [f'<text x="{l}" y="20" class="cap" style="fill:{BLUE}">measured with the reference  ■</text>',
         f'<text x="{l+330}" y="20" class="cap" style="fill:{AQUA}">measured without it  ■</text>']
    for i, (name, with_ref, without_ref, pairs) in enumerate(PEPTIDES):
        y = t + i * (rowh + gap)
        for ln, part in enumerate(name.split("\n")):
            p.append(f'<text x="{l-14}" y="{y+18+ln*14}" class="cap" text-anchor="end">{part}</text>')
        p.append(f'<rect x="{l}" y="{y+2}" width="{X(with_ref)-l:.0f}" height="17" rx="3" fill="{BLUE}" opacity="0.35"/>')
        p.append(f'<rect x="{l}" y="{y+22}" width="{X(without_ref)-l:.0f}" height="17" rx="3" fill="{AQUA}"/>')
        p.append(f'<text x="{X(with_ref)+8:.0f}" y="{y+15}" class="cap">{with_ref:.3f}</text>')
        p.append(f'<text x="{X(without_ref)+8:.0f}" y="{y+35}" class="cap">{without_ref:.3f}</text>')
    return svg(w, h, "".join(p), "effect per peptide")


def chart_frames():
    """Searched frame finds the wrong membrane; reference frame transfers the right one."""
    p = []
    for k, (title, ang, col, note) in enumerate([
            ("we let the maths pick", -62, RED, "66–83° wrong"),
            ("we copy it from an experiment", 0, GREEN, "correct")]):
        ox = 40 + k * 420
        p.append(f'<text x="{ox+170}" y="22" class="cap" text-anchor="middle" style="font-weight:600">{title}</text>')
        cx, cy = ox + 170, 130
        a = math.radians(ang)
        dx, dy = math.cos(a) * 150, math.sin(a) * 150
        for off in (-30, 30):
            nx, ny = -math.sin(a) * off, math.cos(a) * off
            p.append(f'<line x1="{cx-dx+nx:.0f}" y1="{cy-dy+ny:.0f}" x2="{cx+dx+nx:.0f}" y2="{cy+dy+ny:.0f}" '
                     f'stroke="{col}" stroke-width="2.5" opacity="0.55"/>')
        p.append(f'<path d="M{cx-dx:.0f},{cy-dy:.0f} L{cx+dx:.0f},{cy+dy:.0f}" stroke="{col}" '
                 f'stroke-width="1" stroke-dasharray="4 4" opacity="0.5"/>')
        # the peptide: always the same, drawn horizontally
        pts = [(cx - 100 + i * 200/13, cy + 17 * math.sin(i/13*7)) for i in range(14)]
        d = " ".join(f"{'M' if j==0 else 'L'}{x:.0f},{y:.0f}" for j,(x,y) in enumerate(pts))
        p.append(f'<path d="{d}" fill="none" stroke="var(--ink)" stroke-width="3.5" stroke-linecap="round"/>')
        p.append(f'<text x="{cx}" y="236" class="cap" text-anchor="middle" style="fill:{col};font-weight:600">{note}</text>')
    return svg(860, 252, "".join(p), "two ways to place the membrane")


SLIDES = []


def slide(kind, title, body, note=""):
    SLIDES.append((kind, title, body, note))


# ----------------------------------------------------------------------------------
slide("title", "Steering a learned sampler toward a membrane",
      """<p class="sub">Phase 1 · guiding a diffusion model on R<sup>3</sup> × SO(3) with a
      depth-dependent free energy</p>
      <p class="meta">BioEmu · inference-time guidance · melittin, pardaxin, magainin-2</p>""")

slide("", "The physical system",
      """<p class="big">A protein is a heteropolymer: ~20 monomer types, one linear chain.</p>
      <ul><li>Backbone bond lengths and angles are <strong>stiff</strong> — effectively holonomic
      constraints. The soft degrees of freedom are torsions.</li>
      <li>So the configuration space is a constrained manifold, not R<sup>3n</sup>.</li>
      <li>Each monomer carries a side group: some nonpolar, some charged.</li></ul>
      <p class="note">That last point is the entire driving force for what follows.</p>""")

slide("", "The second environment",
      """<p class="big">A cell membrane is a ~30 Å slab of hydrocarbon in water.</p>
      <ul><li>Nonpolar side groups lower their free energy <strong>inside</strong> the slab.</li>
      <li>Charged ones pay a large desolvation penalty there and stay <strong>outside</strong>.</li>
      <li>A chain carrying both is frustrated: it must choose an orientation and a depth.</li></ul>
      <p class="note">Short peptides are often disordered in bulk water and order only at the
      interface. The interesting physics is the <em>transition</em>.</p>""")

slide("math", "What we actually want",
      """<div class="eqbox">
      <div class="eq">x ~ p(x) &nbsp;∝&nbsp; exp( −U(x) / k<sub>B</sub>T )</div>
      <p class="eqcap">samples from the equilibrium ensemble over conformations x</p>
      </div>
      <ul><li>Not one minimum — the <strong>distribution</strong>. Function often lives in the tail.</li>
      <li>Molecular dynamics integrates the Langevin equation honestly, but barrier crossings are
      rare events: µs of simulation for ms of physics.</li></ul>
      <p class="note">So: can a generative model act as a sampler for p(x), trained once,
      amortised over all queries?</p>""")

slide("chart", "Diffusion models as learned samplers",
      f"""<div class="fig">{chart_diffusion()}</div>
      <p class="note">Noise the data to a tractable prior, then learn to reverse it. The reverse
      pass is an SDE integrated in 100 steps. BioEmu is this, trained on ~200 ms of aggregate MD
      plus experimental folding free energies.</p>""")

slide("math", "Forward: Ornstein–Uhlenbeck to a Gaussian prior",
      """<div class="eqbox">
      <div class="eq">dx &nbsp;=&nbsp; −½ β(t) x &nbsp;dt&nbsp; + &nbsp;√β(t) &nbsp;dW</div>
      <p class="eqcap">variance-preserving; p<sub>T</sub>(x) → N(0, I)</p>
      </div>
      <div class="eqbox">
      <div class="eq">x<sub>t</sub> &nbsp;=&nbsp; α<sub>t</sub> x<sub>0</sub> &nbsp;+&nbsp;
      σ<sub>t</sub> ε , &nbsp;&nbsp; ε ~ N(0, I) , &nbsp;&nbsp; α<sub>t</sub><sup>2</sup> +
      σ<sub>t</sub><sup>2</sup> = 1</div>
      <p class="eqcap">the transition kernel is Gaussian in closed form — this is what makes
      training cheap</p>
      </div>""")

slide("math", "Reverse: Anderson's time-reversed SDE",
      """<div class="eqbox">
      <div class="eq">dx &nbsp;=&nbsp; [ −½β(t) x &nbsp;−&nbsp; β(t) <span class="hl">∇<sub>x</sub>
      log p<sub>t</sub>(x)</span> ] dt &nbsp;+&nbsp; √β(t) &nbsp;dW̄</div>
      <p class="eqcap">Anderson 1982 — exact, given the score of every marginal</p>
      </div>
      <p class="big">The <span class="hl">score</span> is the only unknown.</p>
      <div class="eqbox">
      <div class="eq">p ∝ e<sup>−U/k<sub>B</sub>T</sup> &nbsp;&nbsp;⟹&nbsp;&nbsp;
      ∇ log p &nbsp;=&nbsp; −∇U / k<sub>B</sub>T &nbsp;=&nbsp; F / k<sub>B</sub>T</div>
      <p class="eqcap">for a Boltzmann distribution the score <em>is</em> the force field, in units of k<sub>B</sub>T</p>
      </div>""")

slide("math", "Training: denoising score matching",
      """<div class="eqbox">
      <div class="eq">L(θ) &nbsp;=&nbsp; E<sub>t, x<sub>0</sub>, ε</sub>
      ‖ s<sub>θ</sub>(x<sub>t</sub>, t) &nbsp;+&nbsp; ε / σ<sub>t</sub> ‖<sup>2</sup></div>
      <p class="eqcap">because ∇<sub>x<sub>t</sub></sub> log p(x<sub>t</sub> | x<sub>0</sub>) =
      −ε / σ<sub>t</sub> for the Gaussian kernel above</p>
      </div>
      <p class="note">Regression against a known analytic target. No partition function, no
      normalisation, nothing to integrate. This is why it scales where free-energy methods struggle.</p>""")

slide("math", "The state space is not R<sup>3n</sup>",
      """<p class="big">BioEmu puts a rigid frame on every monomer.</p>
      <div class="codebox">positions&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x ∈ R<sup>n×3</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;one point per monomer
orientations&nbsp;&nbsp;R ∈ SO(3)<sup>n</sup>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;how each frame is rotated</div>
      <ul><li>Two coupled diffusions: Gaussian on R<sup>3</sup>, <strong>IGSO(3)</strong> — the heat
      kernel on the group — on the rotations.</li>
      <li>The SO(3) score lives in the Lie algebra so(3) ≅ R<sup>3</sup>.</li>
      <li>All other atoms are <strong>reconstructed</strong> from (x, R) assuming ideal local geometry.</li></ul>
      <p class="note">Nothing in the model <em>enforces</em> that neighbouring frames join up. It has
      only learned to emit consistent ones. Hold that thought.</p>""")

slide("", "Conditioning versus guidance",
      """<p class="big">Two different ways to bias a generative model.</p>
      <ul><li><strong>Conditioning</strong> is trained in. BioEmu learns p(x | c) where c is a
      representation of the sequence from AlphaFold2. Changing c means retraining.</li>
      <li><strong>Guidance</strong> is applied at inference. It modifies the sampled distribution
      using an external function, with the weights frozen.</li></ul>
      <p class="note">We have no membrane training data and no budget to retrain. Guidance is the
      only lever — and because the score is a gradient, it is a clean one.</p>""")

slide("math", "Guidance = tilting the target",
      """<div class="eqbox">
      <div class="eq">p̃(x) &nbsp;∝&nbsp; p(x) &nbsp;·&nbsp; e<sup>−λ E(x)</sup>
      &nbsp;&nbsp;⟹&nbsp;&nbsp; ∇ log p̃ &nbsp;=&nbsp; ∇ log p &nbsp;−&nbsp; λ ∇E</div>
      <p class="eqcap">an exponential tilt of the learned measure; the scores simply add</p>
      </div>
      <div class="eqbox">
      <div class="eq">s<sub>guided</sub> &nbsp;=&nbsp; s<sub>θ</sub>(x<sub>t</sub>, t)
      &nbsp;−&nbsp; <span class="hl">λ ∇<sub>x</sub> E</span></div>
      </div>
      <p class="note">At λ = 1/k<sub>B</sub>T this is exactly adding E to the potential:
      p̃ ∝ exp(−(U+E)/k<sub>B</sub>T). Any other λ is a deliberate over- or under-weighting.</p>""")

slide("math", "The obstruction: E needs a clean configuration",
      """<p class="big">∇E must be evaluated somewhere. Mid-trajectory, x<sub>t</sub> is noise.</p>
      <div class="eqbox">
      <div class="eq">x̂<sub>0</sub> &nbsp;=&nbsp; E[ x<sub>0</sub> | x<sub>t</sub> ] &nbsp;=&nbsp;
      <span class="frac"><span>x<sub>t</sub> + σ<sub>t</sub><sup>2</sup> s<sub>θ</sub>(x<sub>t</sub>, t)</span><span>α<sub>t</sub></span></span></div>
      <p class="eqcap">Tweedie's formula — the posterior mean, free from the score we already have</p>
      </div>
      <p class="note">We score x̂<sub>0</sub> and backpropagate through it. Exact as a posterior
      mean; a poor proxy for a <em>sample</em> when σ<sub>t</sub> is large. This approximation
      caused one of our two failures.</p>""")

slide("chart", "The potential: depth-dependent transfer free energy",
      f"""<div class="fig">{chart_ez()}</div>
      <p class="note">Senes et al. 2007: per-monomer-type ΔG of transfer, fitted against depth in
      known membrane structures. Nonpolar monomers gain ~0.6 kcal/mol in the core; charged ones pay
      ~1.7. Sum over the chain → one scalar. Note it is <strong>separable</strong> in the monomers.</p>""")

slide("", "A gauge problem",
      """<p class="big">p(x) is SE(3)-invariant. E(x) is not.</p>
      <ul><li>The generator has no preferred frame — rotate a sample and it is equally valid.</li>
      <li>But "depth" is defined against an <strong>external</strong> slab, with a normal
      <strong>n</strong> and an offset c.</li>
      <li>So guidance requires <em>fixing the gauge</em>, and how we fix it is a modelling choice,
      not a detail.</li></ul>
      <p class="note">Two options: determine (n, c) variationally, or transport it from a structure
      where it is already known. We chose the first. It was the central error of Phase 1.</p>""")

slide("chart", "Why the variational gauge fails",
      f"""<div class="fig">{chart_frames()}</div>
      <p class="note">Minimising E over (n, c) as well as x lets the objective relocate the slab
      instead of improving the configuration. Benchmarked against an experimentally determined
      placement, the recovered normal was <strong>66–83° off</strong>.</p>""")

slide("bad", "Attempt 1 — the ensemble collapsed",
      """<table class="t"><tr><th>λ</th><th>valid configurations</th><th>⟨E⟩</th></tr>
      <tr><td>0</td><td class="ok">94%</td><td>−2.68</td></tr>
      <tr><td>100</td><td class="warn">72%</td><td>−2.84</td></tr>
      <tr><td>500</td><td class="bad">4%</td><td>−1.31</td></tr></table>
      <p class="note">⟨E⟩ got <em>worse</em> at strong coupling while the chain fell apart. Diagnostic:
      configurations failing the validity check scored <strong>better</strong> than those passing it
      (−3.18 vs −2.84). The tilt was being satisfied by breaking the polymer.</p>""")

slide("math", "Why it was exploitable",
      """<div class="eqbox">
      <div class="eq">E &nbsp;=&nbsp; Σ<sub>i</sub> &nbsp;f( type<sub>i</sub> , n·x<sub>i</sub> − c )</div>
      <p class="eqcap">separable: no two-body term, no chain connectivity</p>
      </div>
      <ul><li>Its infimum puts nonpolar monomers at the midplane and charged ones at infinity
      <strong>simultaneously</strong> — while bonded 3.8 Å apart.</li>
      <li>Unreachable for a connected chain, so the gradient points permanently at a configuration
      that does not exist.</li></ul>
      <p class="note">For melittin: infimum −6.64 kcal/mol; best an ideal helix reaches −2.66;
      the unguided ensemble already sits at −2.68. No headroom without breaking bonds.</p>""")

slide("", "So we built a screen for the objective",
      """<p class="big">Validate the potential before coupling it to the sampler.</p>
      <ul><li><strong>Order check</strong> — does E correlate negatively with helicity? It must.</li>
      <li><strong>Adversarial check</strong> — apply a transform known to be invalid (dilate the
      chain along <strong>n</strong>). E must increase.</li>
      <li><strong>Gauge check</strong> — hold a known structure fixed, sample 4,000 random (n, c).
      The experimental placement must rank at the top.</li></ul>
      <p class="note">Nine candidate potentials. Seven were exploitable under a variational gauge.</p>""")

slide("good", "Attempt 2 — transport the gauge instead",
      """<p class="big">Take (n, c) from an experimentally placed structure by Kabsch superposition.</p>
      <table class="t"><tr><th>potential</th><th>corr. with order</th><th>exploitable</th></tr>
      <tr><td>Ez, variational gauge</td><td class="bad">+0.52</td><td class="bad">yes, −1.98σ</td></tr>
      <tr><td>Ez, transported gauge</td><td class="ok">−0.56</td><td class="ok">no, +0.78σ</td></tr></table>
      <p class="note">All nine potentials flipped to the correct sign. The potential was never the
      main defect — <strong>the gauge was</strong>. The unmodified Ez turned out to be the best of
      the nine.</p>""")

slide("bad", "Attempt 2 still broke the chain",
      """<p class="big">Correct potential, correct gauge, 98% invalid.</p>
      <ul><li>Failures were <strong>always</strong> in the inter-monomer bond (0.74–0.98), never
      steric overlap (0.04–0.10).</li>
      <li>We apply ∇E to <strong>x</strong> and never to <strong>R</strong>.</li>
      <li>The two channels decohere, and the reconstructed bond — which depends on both — snaps.</li></ul>
      <p class="note">Back to the state-space slide: nothing enforces frame consistency. We were
      destroying the one property the model held only statistically.</p>""")

slide("math", "And an error of our own",
      """<p class="big">Our estimator was throwing away the signal.</p>
      <ul><li>Runs share a seed, so sample j with and without guidance share a prior draw:
      <strong>matched pairs</strong>, correlation ρ = 0.994.</li></ul>
      <div class="eqbox">
      <div class="eq">Var(g − b) = Var(g) + Var(b) − 2ρ√(Var g · Var b)
      &nbsp;=&nbsp; 2.16 &nbsp;−&nbsp; 2.13 &nbsp;=&nbsp; 0.03</div>
      <p class="eqcap">pairing cancels 99% of the variance — a control variate, effectively</p>
      </div>
      <table class="t"><tr><th>same data, estimator</th><th>p</th></tr>
      <tr><td>unpaired</td><td class="bad">0.43</td></tr>
      <tr><td>paired</td><td class="ok">4.3 × 10<sup>−8</sup></td></tr></table>""")

slide("good", "The effect is real",
      """<p class="hero">44 / 44</p>
      <p class="big">Every configuration improved, at zero cost to validity.</p>
      <ul><li>94% valid — identical to the unguided ensemble.</li>
      <li>Reproduced on three further peptides, all significant.</li></ul>""")

slide("chart", "A fixed exchange rate",
      f"""<div class="fig">{chart_lambda()}</div>
      <p class="note">⟨ΔE⟩ grows with λ; the surviving fraction falls; their <strong>product is
      constant to within 13% over a 25× range of λ</strong>. The points lie on one hyperbola.
      That is the signature of a conserved trade-off, not a tuning problem.</p>""")

slide("", "A controlled comparison",
      """<p class="big">One sequence, two experimentally determined membrane states.</p>
      <ul><li><strong>Pardaxin</strong> is solved both surface-adsorbed and inserted.</li>
      <li>Same sequence ⇒ same conditioning ⇒ same prior ensemble. Only the target gauge differs.</li>
      <li>Surface target: ⟨ΔE⟩ = −0.266, 48/50 retained. Inserted target: −0.173, 40/50.</li></ul>
      <p class="note">Adsorption is an easier tilt than insertion — consistent with the ceiling,
      since insertion demands the larger displacement of the chain.</p>""")

slide("chart", "Leakage, and the honest effect size",
      f"""<div class="fig">{chart_peptides()}</div>
      <p class="note">Pale = evaluated in the <em>same</em> transported gauge the guidance used.
      Solid = evaluated with an independently determined gauge. The effect is <strong>3–8×
      smaller</strong> when the estimator does not share information with the objective. Still
      significant (p = 0.003–0.03); magnitude ≈ 0.3 k<sub>B</sub>T.</p>""")

slide("", "What Phase 1 established",
      """<ul><li><strong>Inference-time guidance of a frozen sampler works</strong> — monotone in λ,
      100% of configurations, four peptides.</li>
      <li><strong>Gauge fixing dominates potential design.</strong> The opposite of our prior.</li>
      <li><strong>Separable potentials are exploitable</strong> by construction, and must be screened
      adversarially before use.</li>
      <li><strong>A conserved effect-versus-validity trade-off</strong> bounds what guiding x alone
      can achieve.</li></ul>""")

slide("", "Phase 2",
      """<ul><li><strong>Guide the SO(3) channel.</strong> Rotating a frame moves side groups without
      displacing any backbone point, so it cannot violate the bond constraint. If the trade-off
      slope differs, the position channel was the binding one.</li>
      <li><strong>Screen on noised intermediates</strong>, not just x<sub>0</sub> — the gap that let
      two failures through.</li>
      <li><strong>Hold out the gauge</strong> used for evaluation.</li></ul>
      <p class="note">Falsifiable either way: SO(3) guidance has non-zero gradient only for an
      orientation-dependent potential, which also tests whether depth alone is the right order
      parameter.</p>""")

slide("end", "Thank you", """<p class="sub">Questions</p>""")


def main():
    body = ""
    total = len(SLIDES)
    for i, (kind, title, html, note) in enumerate(SLIDES):
        cls = f"slide {kind}".strip()
        head = f"<h2>{title}</h2>" if kind not in ("title", "end") else f"<h1>{title}</h1>"
        body += (f'<section class="{cls}" data-n="{i}">'
                 f'<div class="inner">{head}{html}</div>'
                 f'<div class="pg">{i+1} / {total}</div></section>')

    html = f"""<meta name="claude:viewport-fit-sides" content="cover">
<title>Phase 1 Thesis Deck</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one 16:9 slide at a time, arrow keys to move */
:root {{
  --bg:#fbfbfa; --panel:#ffffff; --ink:#151917; --ink-2:#4a514a; --ink-3:#7d837a;
  --line:#e3e5df; --accent:#1f6f6b; --core:#eef3f1;
  --ok:{GREEN}; --bad:{RED}; --warn:{YELLOW};
  --f-disp:'Space Grotesk',system-ui,sans-serif;
  --f-body:'Inter',system-ui,sans-serif;
  --f-mono:'IBM Plex Mono',ui-monospace,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#101211; --panel:#191c1a; --ink:#f2f4ef; --ink-2:#b9c0b7; --ink-3:#848b82;
    --line:#2b302c; --accent:#63bdb5; --core:#1d2321; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#101211; --panel:#191c1a; --ink:#f2f4ef; --ink-2:#b9c0b7; --ink-3:#848b82;
  --line:#2b302c; --accent:#63bdb5; --core:#1d2321; color-scheme: dark;
}}
html,body {{ height:100%; }}
body {{ background:var(--bg); color:var(--ink); font-family:var(--f-body); margin:0; }}
.slide {{ display:none; min-height:100%; box-sizing:border-box;
  padding-left:max(16px, env(safe-area-inset-left, 0px));
  padding-right:max(16px, env(safe-area-inset-right, 0px));
  padding-block:44px 64px; }}
.slide.on {{ display:flex; flex-direction:column; justify-content:center; }}
.inner {{ max-width:1000px; margin:0 auto; width:100%; }}
h1 {{ font-family:var(--f-disp); font-size:clamp(30px,5.4vw,56px); line-height:1.08;
  margin:0 0 20px; letter-spacing:-0.025em; text-wrap:balance; }}
h2 {{ font-family:var(--f-disp); font-size:clamp(23px,3.6vw,38px); line-height:1.14;
  margin:0 0 24px; letter-spacing:-0.02em; text-wrap:balance; }}
h2::after {{ content:""; display:block; width:54px; height:3px; background:var(--accent);
  margin-top:14px; border-radius:2px; }}
.big {{ font-size:clamp(18px,2.4vw,26px); color:var(--ink); font-weight:500; margin:0 0 20px;
  max-width:30ch; line-height:1.35; }}
.hero {{ font-family:var(--f-disp); font-size:clamp(64px,13vw,150px); color:var(--ok);
  margin:0 0 4px; line-height:1; letter-spacing:-0.04em; }}
ul {{ margin:0 0 14px; padding-left:24px; }}
li {{ font-size:clamp(15px,1.75vw,20px); margin-bottom:13px; line-height:1.5; max-width:46ch;
  color:var(--ink-2); }}
li strong, li em {{ color:var(--ink); }}
.sub {{ font-size:clamp(16px,2.1vw,23px); color:var(--ink-2); margin:0 0 8px; }}
.meta {{ font-family:var(--f-mono); font-size:13px; color:var(--ink-3); margin:0; }}
.note {{ font-size:clamp(13px,1.5vw,16px); color:var(--ink-3); max-width:60ch;
  border-left:3px solid var(--accent); padding-left:14px; margin:18px 0 0; line-height:1.5; }}
.note strong {{ color:var(--ink-2); }}
.fig {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:16px; margin:0 0 6px; }}
.cap {{ font-size:12.5px; fill:var(--ink-2); font-family:var(--f-body); }}
.eqbox {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:18px 22px; margin:0 0 14px; overflow-x:auto; }}
.eq {{ font-family:var(--f-mono); font-size:clamp(15px,2.1vw,25px); color:var(--ink);
  white-space:nowrap; }}
.eqcap {{ font-size:13px; color:var(--ink-3); margin:10px 0 0; }}
.hl {{ color:var(--accent); font-weight:600; }}
.frac {{ display:inline-flex; flex-direction:column; vertical-align:middle; text-align:center;
  margin:0 4px; }}
.frac span:first-child {{ border-bottom:1.6px solid var(--ink); padding:0 8px 3px; }}
.frac span:last-child {{ padding:3px 8px 0; }}
.codebox {{ font-family:var(--f-mono); font-size:clamp(12px,1.5vw,16px); background:var(--core);
  border-radius:8px; padding:16px 18px; margin:0 0 18px; white-space:pre; overflow-x:auto;
  color:var(--ink-2); line-height:1.7; }}
table.t {{ border-collapse:collapse; margin:4px 0 8px; font-size:clamp(14px,1.7vw,19px); }}
table.t th {{ font-family:var(--f-disp); font-size:12px; letter-spacing:0.06em; text-transform:uppercase;
  color:var(--ink-3); text-align:left; padding:0 30px 9px 0; font-weight:600; white-space:nowrap; }}
table.t td {{ padding:10px 30px 10px 0; border-top:1px solid var(--line); color:var(--ink-2);
  font-variant-numeric:tabular-nums; white-space:nowrap; }}
td.ok {{ color:var(--ok); font-weight:600; }}
td.bad {{ color:var(--bad); font-weight:600; }}
td.warn {{ color:var(--warn); font-weight:600; }}
.pg {{ position:fixed; right:max(18px, env(safe-area-inset-right, 0px));
  bottom:calc(16px + env(safe-area-inset-bottom, 0px)); font-family:var(--f-mono);
  font-size:12px; color:var(--ink-3); }}
.slide.title h1, .slide.end h1 {{ color:var(--accent); }}
.bar {{ position:fixed; left:0; bottom:0; height:3px; background:var(--accent); z-index:5;
  transition:width .18s ease; }}
.hint {{ position:fixed; left:max(18px, env(safe-area-inset-left, 0px));
  bottom:calc(16px + env(safe-area-inset-bottom, 0px)); font-size:12px; color:var(--ink-3); }}
@media (prefers-reduced-motion: reduce) {{ .bar {{ transition:none; }} }}
</style>

{body}
<div class="bar" id="bar"></div>
<div class="hint">← → to move</div>
<script>
(function () {{
  var s = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var i = 0;
  function show(n) {{
    i = Math.max(0, Math.min(s.length - 1, n));
    s.forEach(function (el, k) {{ el.classList.toggle('on', k === i); }});
    document.getElementById('bar').style.width = ((i + 1) / s.length * 100) + '%';
    try {{ history.replaceState(null, '', '#s' + (i + 1)); }} catch (e) {{}}
  }}
  document.addEventListener('keydown', function (e) {{
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{ show(i + 1); e.preventDefault(); }}
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{ show(i - 1); e.preventDefault(); }}
    if (e.key === 'Home') show(0);
    if (e.key === 'End') show(s.length - 1);
  }});
  document.addEventListener('click', function (e) {{
    if (e.target.closest('a')) return;
    show(i + (e.clientX > window.innerWidth * 0.35 ? 1 : -1));
  }});
  var m = (location.hash || '').match(/^#s(\\d+)$/);
  show(m ? parseInt(m[1], 10) - 1 : 0);
}})();
</script>
"""
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html)/1024:.0f} KB, {total} slides)")


if __name__ == "__main__":
    main()
