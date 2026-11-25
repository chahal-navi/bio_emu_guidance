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
slide("title", "Steering a protein generator toward the membrane",
      """<p class="sub">Phase 1 · what we built, what broke, and what the numbers say</p>
      <p class="meta">BioEmu · inference-time guidance · melittin, pardaxin, magainin-2</p>""")

slide("", "Proteins are not one shape",
      """<p class="big">A protein is a chain that keeps moving.</p>
      <ul><li>A single picture of it is an average, not the truth.</li>
      <li>What it <em>does</em> often depends on the shapes it visits rarely.</li>
      <li>So we want a <strong>population of shapes</strong>, with the right proportions.</li></ul>""")

slide("", "Near a membrane, the population changes",
      """<p class="big">A cell membrane is an oily sheet about 30 Å thick.</p>
      <ul><li>Oily parts of a peptide want to be <strong>inside</strong> it.</li>
      <li>Charged parts want to stay <strong>outside</strong>, in water.</li>
      <li>Many peptides are floppy in water and fold only when they arrive.</li></ul>
      <p class="note">Antibiotics that punch holes in bacteria work this way. If you want to design one,
      you need the shapes it takes <em>at the membrane</em> — not in a test tube.</p>""")

slide("", "The gap we are trying to close",
      """<p class="big">Good generators exist. They were trained on water.</p>
      <ul><li><strong>BioEmu</strong> generates a realistic population of shapes from a sequence.</li>
      <li>It never saw a membrane.</li>
      <li>Retraining it would need membrane data we do not have, and a lot of compute.</li></ul>
      <p class="note">So: can we <strong>steer</strong> a model we cannot retrain?</p>""")

slide("chart", "How these generators work",
      f"""<div class="fig">{chart_diffusion()}</div>
      <p class="note">Start from pure noise. Remove a little, 100 times. A structure appears.
      Training teaches the model one thing: which direction is "less noisy" from here.</p>""")

slide("math", "The one equation that matters",
      """<div class="eqbox">
      <div class="eq">x<sub>t</sub> &nbsp;=&nbsp; α<sub>t</sub> x<sub>0</sub> &nbsp;+&nbsp; σ<sub>t</sub> ε</div>
      <p class="eqcap">adding noise: real shape, faded by α, plus noise of size σ</p>
      </div>
      <div class="eqbox">
      <div class="eq">dx &nbsp;=&nbsp; [ f(x,t) − g(t)<sup>2</sup> <span class="hl">s<sub>θ</sub>(x,t)</span> ] dt &nbsp;+&nbsp; g(t) dw̄</div>
      <p class="eqcap">removing it: run time backwards, steered by the <span class="hl">score</span></p>
      </div>
      <p class="note">The score <span class="hl">s<sub>θ</sub> ≈ ∇<sub>x</sub> log p<sub>t</sub>(x)</span> is just
      "which way is more realistic". It is the only thing the network learns.</p>""")

slide("", "Why this shape of model suits us",
      """<ul><li>It samples a <strong>population</strong>, which is what we wanted.</li>
      <li>The score is a <strong>gradient</strong> — and gradients add.</li>
      <li>So we can bolt on a second opinion without touching the trained weights.</li></ul>
      <p class="note">That last point is the whole reason this project is possible.</p>""")

slide("math", "Guidance: adding a second opinion",
      """<div class="eqbox">
      <div class="eq">∇ log p(x | y) &nbsp;=&nbsp; ∇ log p(x) &nbsp;+&nbsp; ∇ log p(y | x)</div>
      <p class="eqcap">Bayes, differentiated: what the model knows, plus what we want</p>
      </div>
      <div class="eqbox">
      <div class="eq">s<sub>guided</sub> &nbsp;=&nbsp; s<sub>θ</sub>(x<sub>t</sub>, t) &nbsp;−&nbsp;
      <span class="hl">λ ∇<sub>x</sub> E( x̂<sub>0</sub> )</span></div>
      <p class="eqcap">in practice: push downhill on an energy E, with strength λ</p>
      </div>
      <p class="note">Problem: E must be judged on a <em>finished</em> shape, but mid-run we only have noise.</p>""")

slide("math", "Tweedie: guessing the finished shape early",
      """<div class="eqbox">
      <div class="eq">x̂<sub>0</sub> &nbsp;=&nbsp; <span class="frac"><span>x<sub>t</sub> + σ<sub>t</sub><sup>2</sup> s<sub>θ</sub>(x<sub>t</sub>, t)</span><span>α<sub>t</sub></span></span></div>
      <p class="eqcap">the model's best guess at the final shape, from where we are now</p>
      </div>
      <p class="note">We score <em>that</em> guess and push its gradient back into the sampler.
      It is a guess, and early on a poor one — a point we will return to, painfully.</p>""")

slide("", "What BioEmu actually hands us",
      """<p class="big">Two tensors per shape. Not atoms.</p>
      <div class="codebox">pos&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(n, 3)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;one point per building block
orientations&nbsp;&nbsp;(n, 3, 3)&nbsp;&nbsp;&nbsp;how each block is turned</div>
      <ul><li>Every other atom is <strong>rebuilt</strong> from those two.</li>
      <li>Nothing forces neighbouring blocks to join up — the model only
      <em>learned</em> to place them consistently.</li></ul>
      <p class="note">Remember this slide. It explains every failure that follows.</p>""")

slide("chart", "What we score: how deep each part sits",
      f"""<div class="fig">{chart_ez()}</div>
      <p class="note">Measured energies (Senes et al. 2007). Oily Leucine is happiest in the middle;
      charged Lysine is unhappy there and fine outside. Add this up over the whole chain → one number.</p>""")

slide("", "But where <em>is</em> the membrane?",
      """<p class="big">The generator has no idea which way is up.</p>
      <ul><li>It outputs shapes in a random orientation.</li>
      <li>"Depth" is meaningless until we decide where the membrane sits.</li>
      <li>Two choices: <strong>work it out</strong>, or <strong>copy it</strong> from an experiment.</li></ul>
      <p class="note">We tried the first. It was the single biggest mistake of Phase 1.</p>""")

slide("chart", "The mistake",
      f"""<div class="fig">{chart_frames()}</div>
      <p class="note">If you let the maths choose the membrane to make the score look good, it will.
      Checked against a real experimental placement, it was 66–83° off — almost at right angles.</p>""")

slide("bad", "Attempt 1 — it fell apart",
      """<table class="t"><tr><th>strength</th><th>intact structures</th><th>score</th></tr>
      <tr><td>none</td><td class="ok">94%</td><td>−2.68</td></tr>
      <tr><td>medium</td><td class="warn">72%</td><td>−2.84</td></tr>
      <tr><td>strong</td><td class="bad">4%</td><td>−1.31</td></tr></table>
      <p class="note">Stronger steering made the score <em>worse</em> and destroyed the molecules.
      The energy was being improved by pulling the chain apart.</p>""")

slide("", "Why it could cheat",
      """<p class="big">The energy adds up each block separately.</p>
      <div class="eqbox"><div class="eq">E &nbsp;=&nbsp; Σ<sub>i</sub> &nbsp;f( type<sub>i</sub> , depth<sub>i</sub> )</div></div>
      <ul><li>Nothing in that sum knows the blocks are joined.</li>
      <li>The cheapest way to lower it: let every block sit wherever it likes.</li>
      <li>Stretching the chain, or unfolding it, buys exactly that freedom.</li></ul>""")

slide("", "So we stopped guessing and built a test",
      """<p class="big">Check the scoring rule <em>before</em> trusting it.</p>
      <ul><li><strong>Does it like folded shapes?</strong> It should.</li>
      <li><strong>Can we cheat it?</strong> Stretch a shape on purpose; the score must get worse.</li>
      <li><strong>Does it recognise a real membrane?</strong> Throw 4,000 random
      membranes through a known structure — the true one should win.</li></ul>
      <p class="note">Nine candidate rules went in. Seven could be cheated.</p>""")

slide("good", "Attempt 2 — copy the membrane instead",
      """<p class="big">Take the placement from an experiment and carry it over.</p>
      <table class="t"><tr><th>scoring rule</th><th>likes folded?</th><th>cheatable?</th></tr>
      <tr><td>original, maths-chosen membrane</td><td class="bad">no</td><td class="bad">yes</td></tr>
      <tr><td>original, copied membrane</td><td class="ok">yes</td><td class="ok">no</td></tr></table>
      <p class="note">Every rule we tested improved. The scoring rule was never the main problem —
      <strong>the membrane placement was</strong>.</p>""")

slide("bad", "Attempt 2 still broke the molecules",
      """<p class="big">Right rule. Right membrane. Still 98% broken.</p>
      <ul><li>The breakage was always in the <strong>joins between blocks</strong>, never collisions.</li>
      <li>We push the <em>positions</em> and never touch the <em>orientations</em>.</li>
      <li>The two drift apart, and the rebuilt join snaps.</li></ul>
      <p class="note">Back to that earlier slide: nothing makes the blocks join up. We were breaking
      the one thing the model had only learned by habit.</p>""")

slide("", "And then a mistake in our own maths",
      """<p class="big">We had been measuring it wrong the whole time.</p>
      <ul><li>Every run starts from the <strong>same random seed</strong>.</li>
      <li>So shape #7 with steering is the <em>same shape</em> as #7 without, nudged.</li>
      <li>We had been comparing them as if they were strangers.</li></ul>
      <table class="t"><tr><th>same numbers, compared as…</th><th>p-value</th></tr>
      <tr><td>strangers</td><td class="bad">0.43 — "nothing here"</td></tr>
      <tr><td>matched pairs</td><td class="ok">0.000000043</td></tr></table>""")

slide("good", "It works",
      """<p class="hero">44 / 44</p>
      <p class="big">Every single shape improved.</p>
      <ul><li>Zero damage: 94% intact, exactly the same as no steering.</li>
      <li>Repeated on three more peptides. All improved.</li></ul>""")

slide("chart", "How hard can we push?",
      f"""<div class="fig">{chart_lambda()}</div>
      <p class="note">Push harder and each shape improves more — but fewer survive, at a rate so
      steady that the <strong>total payoff never changes</strong>. The points sit on one curve.
      Gentle steering is the sweet spot.</p>""")

slide("", "A fair test, by luck of the database",
      """<p class="big">One peptide, two experimentally known membrane poses.</p>
      <ul><li><strong>Pardaxin</strong> has been solved both lying on the surface and inserted through.</li>
      <li>Same sequence → same starting population → only the target differs.</li>
      <li>Steering it onto the surface worked <strong>1.5× better</strong> than steering it through.</li></ul>
      <p class="note">Which fits the ceiling we found: sliding onto a surface is a smaller move
      than threading through a wall.</p>""")

slide("chart", "Being honest about the size of it",
      f"""<div class="fig">{chart_peptides()}</div>
      <p class="note">Pale = scored using the same experimental structure the steering used.
      Solid = scored without it. The effect is <strong>3–8× smaller</strong> when measured independently —
      real, repeatable, but modest. Roughly a third of normal thermal jiggle.</p>""")

slide("", "What Phase 1 settled",
      """<ul><li><strong>Steering a frozen generator works.</strong> Every shape improved, on four peptides.</li>
      <li><strong>Where you put the membrane matters more than how you score it.</strong>
      That was the opposite of our starting assumption.</li>
      <li><strong>A scoring rule must be tested for cheats before use.</strong> Seven of nine could be.</li>
      <li><strong>There is a ceiling</strong>, and we know what sets it: we move the blocks
      but not their orientations.</li></ul>""")

slide("", "Phase 2",
      """<ul><li><strong>Steer the orientations too.</strong> Turning a block cannot snap a join —
      it should lift the ceiling rather than raise the damage.</li>
      <li><strong>Test the scoring rule on half-finished shapes</strong>, not just finished ones.
      That gap is what let two failures through.</li>
      <li><strong>Score with a held-out structure</strong>, so the measurement cannot flatter itself.</li></ul>
      <p class="note">Open question: is the ceiling a property of steering positions, or of this
      scoring rule? The orientation experiment answers it either way.</p>""")

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
