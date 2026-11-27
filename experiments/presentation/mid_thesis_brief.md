# Mid-thesis presentation brief: membrane-aware guidance of a protein diffusion model

**Give this whole file to Claude and ask it to produce (a) a slide deck and (b) a spoken script.**
Everything needed is here: the story, every measured number, the equations, the audience, and the
traps. Do not invent data. If a number is not in this file, it was not measured.

---

## 1. The ask

| | |
|---|---|
| Occasion | Mid-thesis review, Phase 1 of 2 |
| Length | **20 minutes** of talk, then questions |
| Deliverables | A slide deck **and** a spoken script keyed to the slides |
| Committee | One bioinformatics professor; two physics professors who work in AI |

**What that audience means in practice.** The physicists will follow SDEs, Fokker-Planck, Boltzmann
measures and Monte Carlo without translation, and they will be bored by hand-waving. They will *not*
know what a residue, a side chain or an alpha-helix is. The bioinformatician is the mirror image:
fluent in the biology, needs the generative modelling explained but is comfortable with the idea of
a learned sampler.

So: **explain the biology in physics language, and state the mathematics properly.** Describe a
protein as a constrained heteropolymer, the membrane as a slab with a hydrophobicity profile, and
the goal as sampling a Boltzmann measure. That phrasing serves both halves of the room at once.

**Length discipline.** 20 minutes is roughly 18-22 slides if some are quick. Do not exceed 24.

---

## 2. What a previous attempt got wrong

A first deck was made and rejected as **too sparse for a thesis presentation**. It was built on
"minimal words, big statements". For a committee that fails: they want to see that the work is
rigorous, and sparse slides hide the rigour.

Correct the balance:

- Slides should carry **real numbers, real tables, and labelled axes**, not slogans.
- Every claim on a slide should have its evidence visible on that slide.
- Keep the *prose* short. A dense table is fine; a dense paragraph is not.
- The speech carries nuance and caveats; the slide carries the evidence.

Also avoid: em dashes and en dashes (use commas or colons), more than two hues in any chart, and
decorative graphics that encode nothing.

---

## 3. The one-sentence thesis

> A diffusion model trained only on proteins in water can be steered, at sampling time and without
> retraining, toward membrane-compatible conformations; the hard part is not the energy function but
> fixing the coordinate frame it is measured in.

That reversal is the intellectual content of Phase 1. Build the talk around it.

---

## 4. Story arc, with timings

The narrative has three failures in it. **Keep them.** A committee reads a flawless story as a story
that has been sanitised, and each failure here produced a real finding.

| # | Beat | Time | Why it is in the talk |
|---|---|---|---|
| 1 | Proteins are ensembles; function lives in the distribution | 1:30 | Motivates sampling over prediction |
| 2 | Membranes change the ensemble; we have no membrane-trained generator | 1:30 | Establishes the gap |
| 3 | Diffusion models as learned samplers; forward/reverse SDE; score | 3:00 | The physicists want this stated properly |
| 4 | Conditioning vs guidance; guidance as an exponential tilt | 2:00 | The core method |
| 5 | The energy we steer with: depth-dependent transfer free energy | 1:30 | What "membrane-compatible" means numerically |
| 6 | **The gauge problem**, and failure 1: the variational frame | 2:30 | The central insight of the phase |
| 7 | How we test an energy before trusting it | 1:30 | Methodological contribution |
| 8 | Failure 2: right energy, right frame, broken molecules | 1:30 | Leads to the Phase 2 plan |
| 9 | Failure 3: our own statistics were wrong | 1:30 | The result was hiding in the estimator |
| 10 | It works: the lambda curve and the conserved trade-off | 2:00 | The positive result |
| 11 | Controls: pardaxin two-state, and the leakage correction | 1:30 | Shows the result was stress-tested |
| 12 | What Phase 1 settled; Phase 2 plan | 1:30 | Close |

Total 21:30. Trim beat 3 or 7 if rehearsal runs long.

---

## 5. Content per beat

### Beat 1-2: the problem

- A protein is a linear heteropolymer, about 20 monomer types. Bond lengths and angles are stiff
  enough to act as holonomic constraints, so the soft coordinates are torsions.
- Each monomer carries a side group: some nonpolar, some charged.
- We want samples from `p(x) proportional to exp(-U(x)/kT)` over conformations, not a single
  minimum. Molecular dynamics is honest but barrier crossings are rare events.
- A cell membrane is a slab of hydrocarbon about 30 Angstrom thick, in water. Nonpolar side groups
  lower their free energy inside it; charged ones pay a desolvation penalty and stay out. A chain
  carrying both is frustrated and must choose a depth and an orientation.
- **BioEmu** is a diffusion model that samples protein conformational ensembles from sequence,
  trained on roughly 200 ms of aggregate MD plus experimental folding free energies. It has never
  seen a membrane. Retraining needs data we do not have.

### Beat 3: diffusion models (state the maths)

Forward, variance-preserving Ornstein-Uhlenbeck:

    dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW
    x_t = alpha_t * x_0 + sigma_t * eps,   eps ~ N(0,I),   alpha_t^2 + sigma_t^2 = 1

Reverse (Anderson 1982), exact given the score of every marginal:

    dx = [ -0.5*beta(t)*x - beta(t) * grad_x log p_t(x) ] dt + sqrt(beta(t)) dWbar

The physics hook, worth saying out loud: for a Boltzmann measure the score **is** the force field
in units of kT.

    p proportional to exp(-U/kT)   =>   grad log p = -grad U / kT = F / kT

Training is denoising score matching, a regression against an analytic target:

    L(theta) = E_{t,x0,eps} || s_theta(x_t,t) + eps/sigma_t ||^2

**State space matters for the story.** BioEmu puts a rigid frame on every monomer:

    positions     x in R^(n x 3)
    orientations  R in SO(3)^n

Two coupled diffusions: Gaussian on R^3, and IGSO(3), the heat kernel on the rotation group, with
its score in the Lie algebra so(3) = R^3. All other atoms are **reconstructed** from (x, R) assuming
ideal local geometry. **Nothing enforces that neighbouring frames join up; the network has only
learned to emit consistent ones.** Plant this explicitly, it pays off at beat 8.

### Beat 4: conditioning vs guidance

- **Conditioning** is trained in: BioEmu learns `p(x|c)` with c an AlphaFold2 representation of the
  sequence. Changing c means retraining.
- **Guidance** is applied at sampling time with the weights frozen.

    ptilde(x) proportional to p(x) * exp(-lambda * E(x))
    grad log ptilde = grad log p - lambda * grad E
    s_guided = s_theta(x_t,t) - lambda * grad_x E

At `lambda = 1/kT` this is exactly adding E to the potential. Any other lambda is a deliberate
over- or under-weighting. Say that; the physicists will want to know what lambda means.

The obstruction: E needs a clean configuration, but mid-trajectory x_t is noise. Use Tweedie's
formula, the posterior mean:

    xhat_0 = E[x_0 | x_t] = ( x_t + sigma_t^2 * s_theta(x_t,t) ) / alpha_t

Exact as a posterior mean, a poor proxy for a *sample* when sigma_t is large. **This approximation
caused one of the failures**, so flag it when introducing it.

### Beat 5: the energy

Senes et al. 2007, "Ez": per-monomer-type transfer free energy fitted against depth in known
membrane protein structures. Two functional forms.

    sigmoidal (17 types):  dE(z) = dE0 / ( 1 + (|z|/Zmid)^n )
    gaussian  (Trp, Tyr):  dE(z) = dEmin * exp( -(|z|-Zmin)^2 / (2 sigma^2) )
    Cys: fitted flat at zero

Example parameters to put on a slide (`dE0, Zmid, n`): Leu `-0.64, 17.34, 8.61`;
Lys `+1.66, 11.11, 2.09`. Gaussian (`dEmin, Zmin, sigma`): Trp `-0.85, 11.65, 7.20`.

Total energy is the **sum over monomers**. That separability is the flaw exposed at beat 6.

### Beat 6: the gauge problem and failure 1

`p(x)` is SE(3)-invariant: the generator has no preferred frame. `E(x)` is not: depth is measured
against an external slab with normal **n** and offset **c**. So guidance requires **fixing a gauge**,
and that choice is a modelling decision.

Two options. **Variational**: minimise E over (n,c) as well as x. **Transported**: take (n,c) from a
structure whose membrane placement is already known experimentally, by Kabsch superposition.

We chose variational. It fails, for a reason worth stating plainly: *if you let the objective choose
where the membrane is, it will put it wherever makes the score look good.* Benchmarked against an
experimentally determined placement the recovered normal was **66 to 83 degrees off**.

Failure 1 data is in section 6. The diagnostic that nails it: configurations **failing** the validity
check scored **better** than those passing (-3.18 vs -2.84 at lambda=100), whereas for the unguided
ensemble broken ones scored *worse* (-2.11 vs -2.68). The tilt was being satisfied by breaking the
polymer.

Why it was exploitable, in one line of algebra:

    E = sum_i f( type_i , n.x_i - c )        separable, no two-body term, no connectivity

Its infimum puts nonpolar monomers at the midplane and charged ones at infinity *simultaneously*,
while they are bonded 3.8 Angstrom apart. For melittin: **infimum -6.64 kcal/mol, best an ideal
helix can reach -2.66, and the unguided ensemble already sits at -2.68.** No headroom exists without
breaking bonds. This is a strong slide: three numbers and the argument is finished.

### Beat 7: screening an energy before use

Three tests, all cheap, all run on structures we already had:

1. **Order check.** Does E correlate negatively with helicity across a generated ensemble spanning
   0 to 92 percent helix? The membrane-bound state of these peptides is helical, so it must.
2. **Adversarial check.** Apply a transform known to be invalid: dilate the chain threefold along
   **n**. E must increase. Report in units of ensemble standard deviation.
3. **Gauge check.** Hold a structure with a known placement fixed and sample 4000 random (n,c).
   The experimental placement must rank near the top.

Nine candidate energies were screened. Seven were exploitable under a variational gauge.

**Credit note for the talk:** test 3 was suggested by the supervisor, and it is the sharpest of the
three because it uses experimental ground truth and removes the search algorithm from the question.
Saying so is good practice and costs nothing.

### Beat 8: failure 2

Correct energy, correct gauge, still 98 percent invalid at strong coupling. The failure signature is
specific and it is the whole point: **breakages were always in the bond between monomers (0.74 to
0.98 of structures), essentially never steric overlap (0.04 to 0.10).**

Explanation, calling back to beat 3: guidance applies `grad E` to **x** and never to **R**. The two
channels decohere, and the reconstructed bond, which depends on both, snaps. No choice of energy
fixes this. It is a statement about which channel is being steered.

### Beat 9: failure 3, our own statistics

Every run in a sweep shares a random seed, and batches are seeded as `base_seed + index`. So sample
j with guidance and sample j without start from the **same prior draw**: matched pairs, with measured
correlation **rho = 0.994**. We had been comparing them as independent samples.

    Var(g - b) = Var(g) + Var(b) - 2 rho sqrt(Var g * Var b)
               = 0.9426 + 1.2137 - 2(0.99382)(0.9709)(1.1017)
               = 0.0303

The variances sum to 2.16; the covariance term removes 2.13. Standard error falls 8.4x, from 0.2214
to 0.0263, on identical data. It is a control variate.

    unpaired:  t = -0.788,  p = 0.43
    paired:    t = -6.641,  p = 4.3e-08

This beat is worth its 90 seconds. It is a genuine methodological lesson and the committee will
recognise it.

### Beat 10-11: results and controls

The positive result, the conserved trade-off, the pardaxin control and the leakage correction.
All numbers in section 6.

**Do not oversell.** The honest effect size is the reference-free column: roughly 0.3 kT. Real,
repeatable, modest. Saying so unprompted is worth more than hiding it.

### Beat 12: Phase 2

1. **Guide the SO(3) channel.** Rotating a frame moves side groups without displacing any backbone
   point, so it cannot violate the bond constraint. Falsifiable: if the effect-versus-validity slope
   differs from the position channel, the position channel was the binding constraint.
2. **Screen on noised intermediates**, not just clean structures. Both failures slipped through
   because every screen evaluated finished configurations while guidance acts on `xhat_0` at
   intermediate noise.
3. **Hold out the evaluation gauge**, so the estimator shares no information with the objective.
4. **Move to real targets with MD ground truth.** We have membrane MD for eleven peripheral
   membrane domains (PH, PX, C2, ENTH; 121 to 160 residues per chain). An MD ensemble is the right
   comparison object for a generative model, and splitting it gives an uncontaminated evaluation.

---

## 6. All measured data

Melittin sequence `GIGAVLKVLTTGLPALISWIKRKRQQ` (26). Pardaxin `GFFALIPKIISSPLFKTLLSAVGSALSSSGGQE`
(33). Magainin-2 `GIGKFLHSAKKFGKAFVGEIMNS` (23). BioEmu v1.1, 50 samples per arm, shared seed,
RTX 4090. Energies in kcal/mol; kT at room temperature is about 0.6 kcal/mol.

### 6.1 Melittin baseline (unguided)

    physical fraction 0.94,  helicity 0.584
    Ez under the transported gauge: mean -1.410, sd 1.130

### 6.2 Lambda curve: Ez on positions, transported gauge, melittin

Paired against the shared-seed baseline, scored under the transported gauge, pairs where both
structures pass the validity check.

```json
{
  "columns": ["lambda","pairs","delta_E","won_fraction","p_paired","p_unpaired","physical","helicity"],
  "rows": [
    [20,  44, -0.174, 1.00, 4.27e-08, 0.514, 0.94, 0.592],
    [35,  37, -0.191, 1.00, 1.41e-07, 0.194, 0.78, 0.652],
    [50,  33, -0.213, 1.00, 3.18e-07, 0.133, 0.70, 0.632],
    [70,  32, -0.250, 1.00, 1.22e-07, 0.088, 0.68, 0.637],
    [100, 26, -0.298, 1.00, 1.75e-06, 0.211, 0.54, 0.608],
    [500, 19, -0.415, 1.00, 8.85e-07, 0.456, 0.40, 0.513]
  ]
}
```

**The key derived quantity:** `|delta_E| * (pairs/50)`, the total improvement surviving per 50
samples generated, equals 0.153, 0.141, 0.141, 0.160, 0.155, 0.158. **Constant to within 13 percent
across a 25-fold range of lambda.** Plot delta_E against surviving fraction and the points lie on a
hyperbola. Suggested chart: scatter with a constant-product contour overlaid.

`lambda = 20` is the operating point: it returns 0.029 of effect per structure lost, about twice the
efficiency of every larger lambda, and costs nothing (0.94 physical, identical to baseline).

**Caveat to state:** results at lambda >= 100 are not interpretable in either direction. Filtered
they are survivorship-biased (48 to 62 percent of pairs discarded); unfiltered they are contaminated
by broken structures whose energies are meaningless.

### 6.3 Four peptides at lambda = 20

"With gauge" = scored in the same transported gauge guidance used. "Without gauge" = the same
structures scored with an independently determined frame. The gap is information leakage.

```json
{
  "columns": ["peptide","length","baseline_physical","guided_physical",
              "with_gauge_delta","with_gauge_won","with_gauge_p",
              "without_gauge_delta","without_gauge_won","without_gauge_p"],
  "rows": [
    ["pardaxin surface", 33, 0.96, 0.98, -0.266, 0.98, 1.5e-09, -0.086, 0.75, 3.0e-03],
    ["melittin",         26, 0.94, 0.94, -0.174, 1.00, 4.3e-08, -0.022, 0.75, 8.1e-03],
    ["pardaxin TM",      33, 0.96, 0.80, -0.173, 0.95, 9.3e-07, -0.025, 0.70, 3.0e-02],
    ["magainin-2",       23, 0.94, 0.88, -0.098, 0.93, 1.9e-05, -0.007, 0.67, 6.0e-01]
  ]
}
```

**Two things to say about this table.** First, the effect shrinks 3 to 8 fold when the evaluation
does not share a frame with the objective. Second, **magainin-2 does not survive the honest test
(p = 0.6)**. Three of four do. State that; it is the kind of detail that earns trust.

### 6.4 The pardaxin control

OPM holds two entries with the **identical 33-residue sequence** in two different binding modes:
`2kns` surface-adsorbed (CA z-range +13.2 to +21.8, does not span the midplane) and `1xc0`
transmembrane (z -19.6 to +8.2, spans it). Same sequence means same conditioning means the same
prior ensemble. Only the target gauge differs.

Surface target: -0.266, 48 of 50 pairs retained. Inserted target: -0.173, 40 of 50.
**Adsorption is an easier tilt than insertion**, which fits the ceiling: insertion demands the larger
displacement of the chain.

### 6.5 Gauge A/B across all nine energies

Correlation with helicity (negative is correct) and the adversarial dilation margin in units of
ensemble sd (positive is safe).

```json
{
  "columns": ["energy","r_searched","r_transported","stretch_searched","stretch_transported"],
  "rows": [
    ["Ez on CA",            0.522, -0.559, -1.98,  0.78],
    ["Ez clamped",          0.065, -0.625, -0.51,  1.12],
    ["Ez + centroid anchor",-0.010, -0.559, -1.60,  0.78],
    ["depth contrast",      0.175, -0.447, -9.05,  0.46],
    ["saturating burial",   0.507, -0.555, -1.16,  0.47],
    ["hydrophobic moment", -0.421, -0.421,  null,  null],
    ["moment x normal",    -0.421, -0.677,  0.24, -0.10],
    ["burial + 3x moment", -0.364, -0.508, -0.21,  0.04],
    ["burial + moment",    -0.045, -0.567, -0.86,  0.32]
  ]
}
```

**Every energy improves when the gauge is transported.** Eight of nine flip to the correct sign on
helicity. The worst exploit, -9.05 sd, becomes +0.46. This is the slide that carries the thesis
sentence.

### 6.6 Gauge recognition screen

Hold the experimentally placed structure fixed, sample 4000 random (n,c), ask where the true
placement ranks. True tilt is 64 degrees, reproducing the published 65 to within 1.

```json
{
  "columns": ["energy","true_placement_percentile","gap_to_best_sd","preferred_tilt_deg"],
  "rows": [
    ["Ez on CA",             96.9, 0.26, 10],
    ["saturating burial",    90.5, 0.94,  7],
    ["burial + 3x moment",   90.5, 0.94,  7],
    ["burial + moment",      90.5, 0.94,  7],
    ["moment x normal",      88.4, 0.38, 80],
    ["Ez + centroid anchor", 86.9, 0.33, 21],
    ["Ez clamped",           72.5, 0.70, 61],
    ["depth contrast",       71.4, 1.13,  3]
  ]
}
```

Two findings. **The unmodified Ez energy is the only one ranking the truth in the top 5 percent**;
its poor showing on the helicity screen was entirely a gauge artefact. And **most energies prefer a
near-transmembrane placement (3 to 10 degrees) over the true surface-bound one (64 degrees)**,
because burial is maximised by threading the chain through the slab, which is the wrong binding mode
for this peptide.

### 6.7 Failure 1, for the slide

    lambda     physical    mean E
    0          0.94        -2.68
    100        0.72        -2.84
    500        0.04        -1.31

Under the variational gauge. E gets *worse* at strong coupling while the ensemble collapses.

### 6.8 Failure 2, for the slide

Transported gauge, two energies that passed the screen, at lambda = 20, 100, 500:

    physical fraction falls 0.94 -> 0.24 -> 0.02
    bond breakages 0.74 to 0.98,  steric overlaps 0.04 to 0.10

### 6.9 Cost, if asked

Whole project ran on a rented RTX 4090 for well under 10 USD. A 50-sample guided run on a 26-residue
peptide takes 14 seconds; the baseline takes 100 to 160 because it pays the one-time MSA and
embedding cost. Scaling to the 130-residue MD targets is roughly 25x the attention cost and 10x more
batches.

---

## 7. Charts worth making

Build these from the data above. Two hues maximum: one accent, one red for failure. Label axes with
units.

1. **Ez energy against depth**, one nonpolar and one charged monomer, with the membrane core shaded.
   Compute from the parameters in beat 5. Shows what the energy *is*, in one picture.
2. **Gauge A/B**, grouped bars from 6.5: helicity correlation before and after, nine energies. The
   sign flip across the board is the single most persuasive image in the talk.
3. **The lambda trade-off**, scatter of `|delta_E|` against surviving fraction from 6.2, with a
   constant-product hyperbola overlaid and each point labelled by lambda.
4. **Leakage**, paired bars from 6.3: with gauge versus without, four peptides.
5. **A schematic of the gauge problem**: the same peptide with a slab at the wrong angle and at the
   right one. Simple line drawing, no data.
6. Optional: forward and reverse diffusion as a strip of increasingly ordered chains.

---

## 8. Likely questions, with answers

**"Is 0.3 kT a meaningful effect?"** Honest answer: it is small, it is highly significant, and it is
consistent across every structure. The ceiling analysis says it is near the maximum achievable while
steering positions alone, which is why Phase 2 targets the rotation channel.

**"Why not just retrain on membrane data?"** No dataset of that kind at the needed scale, and
retraining BioEmu is out of reach. Guidance is modular: the same machinery accepts any differentiable
energy, so a better energy drops in without touching the model.

**"How do you know the generated structures are not just copies of the reference?"** Measured:
conformational RMSD to the reference stays 3.8 to 7.4 Angstrom and guidance moves it by at most 0.18.
No collapse. But the energy improvement does correlate with reference similarity (r up to +0.76),
which is exactly why the reference-free evaluation in 6.3 is the number we quote.

**"Why Kabsch rather than predicting the pose?"** Because the variational alternative demonstrably
finds the wrong membrane, 66 to 83 degrees off. Predicting the pose is the better science and needs
the rotation channel first.

**"Does this transfer beyond helical peptides?"** Unknown. All four targets so far are helical and
small. The MD set in Phase 2 is folded domains of 120 to 160 residues, which is the real test.

**"Your lambda >= 100 numbers look impressive."** They are not interpretable; say so before being
asked. Filtered they are survivorship-biased, unfiltered they are contaminated.

---

## 9. Glossary for the mixed room

Use these framings; they work for both halves.

| Term | Say it as |
|---|---|
| residue / monomer | one unit of the chain, carrying a side group |
| conformation | a configuration of the chain |
| helix | a locally ordered, regularly repeating chain geometry |
| hydrophobic | nonpolar; lowers free energy inside the hydrocarbon slab |
| Ez potential | a depth-dependent transfer free energy, fitted empirically |
| OPM | a database giving experimentally anchored membrane placements |
| Kabsch | optimal rigid superposition of two point sets |
| DSSP | an algorithm labelling local chain order from backbone geometry |
| physicality filter | a validity check on bond lengths and steric overlap |

---

## 10. Tone

Confident about the method, plain about the magnitude, and open about the three failures. The
strongest version of this talk is one where the committee can see that every claim was attacked
before it was believed. Two of the three failures were found by attacking our own results, and the
sharpest test in the deck came from the supervisor. Say both.
