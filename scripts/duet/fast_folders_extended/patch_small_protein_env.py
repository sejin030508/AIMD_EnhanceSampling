import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text(encoding="utf-8")
if "PVB_SITE" in s:
    print("already patched"); raise SystemExit(0)
old = 'export PYTHONPATH="$DUET_PROJECT_ROOT/src"'
new = ('# PVB_SITE carries the isolated --target directory holding torch_scatter\n'
       '# and e3nn; without appending it the sampler subprocess loses them,\n'
       '# because this file is re-sourced by the cell runner.\n'
       'export PYTHONPATH="$DUET_PROJECT_ROOT/src${PVB_SITE:+:$PVB_SITE}"')
assert s.count(old) == 1, f"anchor count {s.count(old)}"
p.write_text(s.replace(old, new), encoding="utf-8")
print("patched", p)
