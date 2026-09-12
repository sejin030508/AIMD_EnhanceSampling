# Extended fast-folder benchmark set

Adds **BBL**, **Protein B** and **Homeodomain** to the small-protein transition
pilot that already covers Chignolin, Trp-cage and BBA.

Nothing here changes the sampler, the reward, or the validity gate.  The three
new proteins are pushed through the pilot's own preparation, hashing and
verification path so their numbers land on the same scale as the existing
three.

## Why these three

Villin was dropped: the fast-folding construct is the HP35 Nle/Nle variant and
its `folded.pdb` contains two **NLE** residues, which are outside the standard
20-residue vocabulary used by ConfRover's `aatype`, by `ff14SBonlysc.xml`, and
by the ESM tokenisers.  Substituting them would change the construct away from
the reference MD, so the protein is excluded rather than silently altered.

BBL costs nothing to add.  TPS-DPS publishes its complete asset bundle
(`folded.pdb`, `unfolded.pdb`, `tica_model.pkl`, `pmf.npy`, `xs.npy`, `ys.npy`)
at the same pinned commit the pilot already uses, so it needs no reference-MD
download and no TICA refit, and its THP is directly comparable to the existing
three.  It is therefore also the de-risking case: it exercises the whole
extended path before the 34.5 GB download matters.

Protein B and Homeodomain need a refitted TICA model, which is what most of
this directory is for.

## Provenance

| Asset | Source |
| --- | --- |
| BBL bundle (all six files) | `kiyoung98/tps-dps` @ `61fd65ad2e2f110d65c176a8c8f5c2fe8bdab034` |
| Protein B / Homeodomain `folded.pdb`, `unfolded.pdb` | `PanosAntoniadis/platito` @ `08093663ff07683eed0f5a02654db370d6b2064b`, `data/fast_folders/initial_structures/` |
| Protein B / Homeodomain reference MD | `pub.htmd.org` via `torchmd/torchmd-protein-thermodynamics` |
| Force field | `data/protein.ff14SBonlysc.xml` from the TPS-DPS checkout |

Both PLaTITO pairs are all-atom including hydrogens, carry matching atom counts
between the folded and unfolded member, and contain no non-standard residues,
so they satisfy `assert_same_topology` as-is.

Temperatures come from PLaTITO's `data/fast_folders/temperatures.json`, which
reproduces the Anton reference temperatures: Protein B 340 K, Homeodomain
360 K.  BBL keeps the TPS-DPS default of 300 K.

> The torchmd dataset README advertises `http://pub.htmd.org/...`, which
> returns **403**.  The same objects are served over `https://` and support
> range requests.

## Scripts

| Script | Purpose |
| --- | --- |
| `fetch_reference_md.py` | Resumable download of one reference archive |
| `launch_downloads.sh` | Runs the three downloads in parallel; safe to re-run |
| `inspect_tica_model.py` | Reads settings back off a published `tica_model.pkl` |
| `fit_tica_model.py` | Refits TICA, optionally validating against a published model |
| `patch_pilot_scripts.py` | Makes the pilot's molecule list and config directory selectable |

### TICA settings recovered from the published models

The benchmark ships fitted models but not the fitting script.  Reading the
pickles gives identical settings across all four published proteins —
`dim=2`, `scaling="kinetic_map"`, `epsilon=1e-6` — with only the lag time
differing per protein:

| Protein | lag (frames) |
| --- | --- |
| chignolin | 500 |
| trpcage | 100 |
| bba | 100 |
| bbl | 1 |

Because the lag is hand-picked per protein, a refit for a new protein cannot
simply copy one.  `fit_tica_model.py --scan-lags` reports implied timescales
across candidate lags so the choice can be made on convergence rather than by
eye.

### Validate before trusting a refit

Run the refit on **BBA first**, where a published model exists:

```bash
python fit_tica_model.py \
  --folded-pdb   external/tps-dps/data/bba/folded.pdb \
  --topology     <reference topology> \
  --trajectories <reference xtc files> \
  --lag 100 --output-dir /tmp/bba_refit \
  --validate-against external/tps-dps/data/bba/tica_model.pkl
```

The report gives the absolute correlation of each projected component against
the published model.  A refit that cannot reproduce BBA is not on the same
scale as the existing results, and a THP computed from it would not be
comparable to the pilot's numbers.

### The 0.75 hit threshold does not transfer

`weighted_valid_thp` counts a path when the first two TICA coordinates of its
final frame land within `0.75` of the folded reference.  That radius is tied to
the scale of the published models.  For a refitted protein, express it as the
quantile of the folded-basin distribution that `0.75` corresponds to in BBA,
then apply the same quantile — do not reuse the number.

`pmf.npy` / `xs.npy` / `ys.npy` are a 50x50 grid used only for the PMF overlay
plot; they do not enter the hit decision.

## Running the preparation

BBL, from the official checkout:

```bash
python small_protein_pilot/prepare_small_protein_inputs.py \
  --official-root external/tps-dps \
  --output-root   data/fast_folders_extended/prepared_inputs \
  --config-root   small_protein_pilot/configs_extended \
  --molecules bbl
```

Then generate the ConfRover representation once per sequence and verify:

```bash
python scripts/duet/case_studies/prepare_confrover_repr.py \
  --config small_protein_pilot/configs_extended/bbl_stride16_t32.yaml

python small_protein_pilot/verify_small_protein_setup.py \
  --project-root "$DUET_PROJECT_ROOT" \
  --code-root small_protein_pilot \
  --data-root "$SMALL_PROTEIN_DATA_ROOT" \
  --molecules bbl --configs-subdir configs_extended
```

Protein B and Homeodomain follow the same two commands once their bundles exist,
with `--official-root` pointing at the extended asset root instead.

## Standing caveat

The Bundle A audit found gross peptide C-N violations in every generated frame,
including frames from the official ConfRover forward sampler.  The validity gate
used here is still C-alpha only, so `weighted_valid_thp` for these proteins means
"passed the C-alpha QC and landed inside the TICA target", not "recovered a
physically valid folding transition".  That limit applies to the extended set
exactly as it does to the original three.
