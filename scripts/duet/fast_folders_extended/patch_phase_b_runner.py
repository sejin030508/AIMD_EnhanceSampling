"""Point phase_b_runner at the backend registry instead of the ConfRover builder."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
if "build_adapter" in s:
    print("already patched"); raise SystemExit(0)
backup = p.with_name(p.name + ".orig")
if not backup.exists():
    backup.write_text(s, encoding="utf-8")
for old, new in (
    ("from confmh.duet.runner import build_confrover_adapter",
     "from confmh.duet.runner import build_adapter"),
    ("adapter = build_confrover_adapter(run_cfg)", "adapter = build_adapter(run_cfg)"),
):
    assert s.count(old) == 1, f"anchor count {s.count(old)}: {old!r}"
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
print(f"patched {p} (backup {backup.name})")
