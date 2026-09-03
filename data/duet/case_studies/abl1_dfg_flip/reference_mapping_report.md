# ABL1 DFG-flip reference mapping

- Sampling sequence: `GMSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQ` (287 residues)
- Sampling topology: canonical residue numbers 227–513; chain 0.
- PDB 6XR6/6XR7 chain A uses coordinates 248–534 and maps as canonical = PDB − 19.
- The two sequences overlap exactly for 285 residues; the WE construct adds canonical 227–228
  and omits canonical 514–515 relative to the NMR construct.
- Generated residue indices: `{'Tyr253': 26, 'Lys271': 44, 'Glu286': 59, 'Val299': 72, 'Thr315': 88, 'Ala380': 153, 'Asp381': 154, 'Phe382': 155}`.
- Endpoint reward uses only the final Asp381/Phe382 pseudo-dihedral state.
- DFG-inter contact, event order, salt-bridge profile, and route identity are evaluation-only.
- Endpoint centers: Asp=265.029°, Phe=33.492°.
- Endpoint scales: Asp=31.039°, Phe=18.434°; success distance ≤ 0.8539.
- Reference route event frames: concerted=[394, 670, 558],
  staggered=[289, 614, 341] (Asp complete, Phe complete, DFG-inter contact).

The public archive does not expose independent WT replicate HDF5 files. Route comparisons are
therefore descriptive and unweighted; rate, free-energy, and route-population claims are blocked.
