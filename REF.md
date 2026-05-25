# REF.md — Reference Surgery Action List (R1-m1)

Reviewer R1-m1 flagged 11 misplaced / weak references. The sentence-level
fixes were applied to `main (4) copy.tex` and `supp_material.tex` by Plan
14-17. The `.bib` file (`bib.bib`) lives **Overleaf-side, not in this
repo**, so the bibliography-entry additions and removals listed below have
to be applied manually in Overleaf before resubmission.

Full per-reference rationale and the verbatim before/after sentence
rewrites are in `revision/docs/paper_blocks_refs_methods.md` §PAPER-06.

---

## 1. ADD these `.bib` entries (3 new — paste into `bib.bib`)

```bibtex
@article{havlicek2019supervised,
  title   = {Supervised learning with quantum-enhanced feature spaces},
  author  = {Havl{\'\i}{\v{c}}ek, Vojt{\v{e}}ch and C{\'o}rcoles, Antonio D. and Temme, Kristan and Harrow, Aram W. and Kandala, Abhinav and Chow, Jerry M. and Gambetta, Jay M.},
  journal = {Nature},
  volume  = {567},
  year    = {2019}
}

@article{schuld2019quantum,
  title   = {Quantum Machine Learning in Feature Hilbert Spaces},
  author  = {Schuld, Maria and Killoran, Nathan},
  journal = {Physical Review Letters},
  volume  = {122},
  year    = {2019}
}

@book{rasmussen2006gaussian,
  title     = {Gaussian Processes for Machine Learning},
  author    = {Rasmussen, Carl Edward and Williams, Christopher K. I.},
  publisher = {MIT Press},
  year      = {2006}
}
```

The Bernal et al. AIChE perspective (`bernal2022perspectives`) is already
cited in the manuscript at three locations (R1-m6 / R2-M2). If it is not
yet in `bib.bib`, add this fourth entry as well:

```bibtex
@article{bernal2022perspectives,
  title   = {Perspectives of quantum computing for chemical engineering},
  author  = {Bernal, David E. and Ajagekar, Akshay and Harwood, Stuart M. and Stober, Spencer T. and Trenev, Dimitar and You, Fengqi},
  journal = {AIChE Journal},
  volume  = {68},
  year    = {2022}
}
```

---

## 2. REMOVE these `.bib` entries (if unused elsewhere in the manuscript)

Each of these was attached to a now-rewritten sentence. Before deleting,
check Overleaf with Ctrl-F to confirm no other `\cite{<key>}` exists.

| Old key | What it was | Why removed (R1-m1 reviewer concern) |
|---|---|---|
| `wang2018esrganenhancedsuperresolutiongenerative` | ESRGAN — image super-resolution GAN | Cited in a time-series context; replaced with the already-defined `yoon2019TimeGAN` |
| `chokwitthaya2020applying` | GMM in construction | Sentence claims Gaussian *process* regression; replaced with `rasmussen2006gaussian` |
| (the old [41] key) | Adaptive rolling-median anomaly detection | Already removed; rolling-window claim now relies on the already-defined `dimoudis2023utilizing` |
| `Liu_2019` | Quantum option pricing | Cited as evidence of quantum advantage for bioprocess generation; removed, not replaced (per PAPER-02 no-overclaiming lock) |
| `Stamatopoulos_2020` | Option-pricing on a quantum computer | Same — removed, not replaced |
| `farhi2014quantumapproximateoptimizationalgorithm` | QAOA | Same — removed, not replaced |
| (the old [55]-[57], [59] keys generally) | VQE / option-pricing / QAOA / adversarial-robustness | Reviewer accepted resolution is *removal* of the over-reaching claim, not substitution |

> **Note on [55]-[57], [59]:** the reviewer's resolution is removal of the
> over-reaching quantum-advantage-for-bioprocess claim, not substitution
> with a different not-yet-demonstrated quantum-advantage citation
> (consistent with the R1-M5 no-overclaiming recalibration).

---

## 3. KEEP these `.bib` entries untouched (reviewer-confirmed appropriate)

> R1-m1 explicitly lists these as anchors that **should not be touched**:
> **[21]–[23], [34]–[36], [61]**.

The renumbering after the removals above will of course shift the bracket
numbers — but the underlying `.cite{}` keys for these anchor references
must remain in `bib.bib` and continue to be cited from the same sentences
they currently are.

---

## 4. Manuscript-side sentence rewrites — VERIFY present in Overleaf

These were applied to `main (4) copy.tex` and `supp_material.tex` in this
repo by Plan 14-17 (commit `e7e6329`). Before resubmitting, sync these to
the Overleaf project and confirm the following:

| Section | What to confirm |
|---|---|
| §1.2 (Synthetic Data Generation Approaches) | "Methods such as Gaussian process regression \cite{rasmussen2006gaussian} and multivariate statistical models…" — uses the new GPR key, not the GMM-in-construction one |
| §1.2 | "Machine learning models such as variational autoencoders (VAEs) and generative adversarial networks (GANs), including time-series-specific architectures \cite{yoon2019TimeGAN}, can learn directly from available data…" — uses TimeGAN, not ESRGAN |
| §1.3 (QGANs intro) | "…GANs more broadly, including classical recurrent variants, have also been applied to healthcare time series \cite{esteban2017realvaluedmedicaltimeseries}." — sentence rephrased to credit classical RCGAN, NOT QGAN-for-healthcare |
| §1.3 / §2.4 | `\cite{Mugel2022}` appears only in optimization context, not as QGAN-for-anything-else evidence |
| §2.1 / §2.4 (Quantum ML review) | "Relevant approaches include quantum-enhanced sampling, quantum kernels \cite{havlicek2019supervised, schuld2019quantum}, and variational quantum algorithms…" — Havlíček + Schuld&Killoran cited together for the quantum-kernel background |
| §3.1 / Supp §A.7 (rolling windows) | "Overlapping subsequences of length 10 with stride 2 were extracted using a rolling window approach \cite{dimoudis2023utilizing}." — only `dimoudis2023utilizing` cited; the old [41] key gone |
| Supp §A.2.3 ("Quantum Advantage for Generative Models") | The sentence about quantum interference / optimization landscapes for bioprocess generation has been rewritten to drop the VQE / option-pricing / QAOA / adversarial-robustness citations; the only remaining cite is `\cite{Cerezo_2021}` for the NISQ-device-constraints statement |

---

## 5. Post-edit checks (run Overleaf-side after applying)

1. Compile the manuscript and confirm no `??` undefined-reference warnings.
2. Open the bibliography section and confirm the numbering for the
   surviving anchors lines up with whatever the resubmitted manuscript
   text refers to (the renumbering happens automatically; just sanity-check
   that no orphaned bracket numbers remain in the prose).
3. Confirm the removed keys (`wang2018esrgan…`, `chokwitthaya2020applying`,
   `Liu_2019`, `Stamatopoulos_2020`, `farhi2014quantumapproximateoptimizationalgorithm`,
   the old [41] key) no longer appear in the compiled bibliography.

---

End of REF.md.
