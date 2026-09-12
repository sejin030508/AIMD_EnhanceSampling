
from pathlib import Path
from pymol import cmd

OUTDIR = Path(r"/Users/sejin/Desktop/AIBL/projects/AI_MD_NSMC/data/phase_b_pockets/raw/alignment_prmt5_6uxy")
STATE_LABELS = {'6uxy': 'holo', '7kic': 'apo'}
KEEP_ORGANICS = {}
CLEAR_ORGANICS = set(['7kic'])
PDBS = {
    "6uxy": Path(r"/Users/sejin/Desktop/AIBL/projects/AI_MD_NSMC/data/phase_b_pockets/raw/alignment_prmt5_6uxy/6uxy_raw.pdb"),
    "7kic": Path(r"/Users/sejin/Desktop/AIBL/projects/AI_MD_NSMC/data/phase_b_pockets/raw/alignment_prmt5_6uxy/7kic_raw.pdb"),
}

def clean_object(name):
    cmd.remove(f"{name} and not (polymer or organic)")
    cmd.remove(f"{name} and solvent")

def polymer_chains(name):
    return [chain for chain in cmd.get_chains(f"{name} and polymer") if chain]

def organic_count(name):
    return cmd.count_atoms(f"{name} and organic")

def organic_resn(name):
    model = cmd.get_model(f"{name} and organic")
    return sorted(set(atom.resn for atom in model.atom))

def relevant_organic_selection(name, chain):
    polymer = f"({name} and polymer and chain {chain})"
    same_chain = f"({name} and organic and chain {chain})"
    nearby = f"byres ({name} and organic within 8 of {polymer})"
    return f"({same_chain} or {nearby})"

def keep_chain_and_relevant_organic(name, chain):
    polymer = f"({name} and polymer and chain {chain})"
    organic = relevant_organic_selection(name, chain)
    cmd.remove(f"{name} and not ({polymer} or {organic})")

def organic_policy(name):
    if name in CLEAR_ORGANICS:
        return "clear"
    if name in KEEP_ORGANICS:
        return f"keep {KEEP_ORGANICS[name]}"
    return "all"

def apply_organic_policy(name):
    if name in CLEAR_ORGANICS:
        before = organic_resn(name)
        cmd.remove(f"{name} and organic")
        return before

    residues = KEEP_ORGANICS.get(name)
    if not residues:
        return []
    before = organic_resn(name)
    keep_expr = "+".join(residues)
    cmd.remove(f"{name} and organic and not resn {keep_expr}")
    after = organic_resn(name)
    removed = sorted(set(before) - set(after))
    return removed

def save_by_state(name):
    suffix = STATE_LABELS.get(name)
    if suffix is None:
        suffix = "holo" if organic_count(name) > 0 else "apo"
    output = OUTDIR / f"{name}_{suffix}.pdb"
    cmd.save(str(output), name)
    return output

cmd.reinitialize()

for pdb_id, path in PDBS.items():
    cmd.load(str(path), pdb_id)
    clean_object(pdb_id)

fixed_id = "6uxy"
mobile_id = "7kic"

fixed_chains = polymer_chains(fixed_id)
mobile_chains = polymer_chains(mobile_id)

print(f"{fixed_id} polymer chains after clean: {fixed_chains}")
print(f"{mobile_id} polymer chains after clean: {mobile_chains}")
print(f"{fixed_id} organic atoms after clean: {organic_count(fixed_id)}")
print(f"{mobile_id} organic atoms after clean: {organic_count(mobile_id)}")

candidates = []

for fixed_chain in fixed_chains:
    for mobile_chain in mobile_chains:
        trial = f"trial_{mobile_chain}_to_{fixed_chain}"
        cmd.create(trial, mobile_id)
        result = cmd.align(
            f"{trial} and polymer and chain {mobile_chain}",
            f"{fixed_id} and polymer and chain {fixed_chain}",
            cycles=5,
            transform=1,
            quiet=1,
        )
        rmsd = float(result[0])
        aligned_atoms = int(result[1])
        candidates.append((aligned_atoms, rmsd, fixed_chain, mobile_chain, trial))
        print(
            f"candidate {mobile_id}:{mobile_chain} -> {fixed_id}:{fixed_chain}, "
            f"aligned_atoms={aligned_atoms}, rmsd={rmsd:.4f}"
        )

if not candidates:
    raise RuntimeError("No polymer chain pairs were available for alignment.")

candidates.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
aligned_atoms, rmsd, fixed_chain, mobile_chain, best_trial = candidates[0]

print(
    f"selected {mobile_id}:{mobile_chain} -> {fixed_id}:{fixed_chain}, "
    f"aligned_atoms={aligned_atoms}, rmsd={rmsd:.4f}"
)

cmd.delete(mobile_id)
cmd.set_name(best_trial, mobile_id)

for _, _, _, _, trial in candidates[1:]:
    if trial != best_trial:
        cmd.delete(trial)

keep_chain_and_relevant_organic(fixed_id, fixed_chain)
keep_chain_and_relevant_organic(mobile_id, mobile_chain)

fixed_removed = apply_organic_policy(fixed_id)
mobile_removed = apply_organic_policy(mobile_id)

fixed_out = save_by_state(fixed_id)
mobile_out = save_by_state(mobile_id)

print(f"final {fixed_id} chain kept: {fixed_chain}")
print(f"final {mobile_id} chain kept: {mobile_chain}")
print(f"requested {fixed_id} state label: {STATE_LABELS.get(fixed_id, 'auto')}")
print(f"requested {mobile_id} state label: {STATE_LABELS.get(mobile_id, 'auto')}")
print(f"requested {fixed_id} organic policy: {organic_policy(fixed_id)}")
print(f"requested {mobile_id} organic policy: {organic_policy(mobile_id)}")
print(f"removed {fixed_id} organic residues: {fixed_removed}")
print(f"removed {mobile_id} organic residues: {mobile_removed}")
print(f"final {fixed_id} organic residues: {organic_resn(fixed_id)}")
print(f"final {mobile_id} organic residues: {organic_resn(mobile_id)}")
print(f"saved {fixed_out}")
print(f"saved {mobile_out}")
