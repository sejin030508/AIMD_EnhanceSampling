import json

from confmh.discovery.analysis import aggregate_alternate_runs


def _summary(start_id: str, target_id: str) -> dict:
    return {
        "model_backend": "confrover",
        "protocol": "target_blind",
        "controller": "biased",
        "category": "domainmotion",
        "test_case": "SHARED_ACCESSION",
        "start_id": start_id,
        "target_id": target_id,
        "valid_hit_rate_3a": 0.1,
        "valid_hit_rate_half_endpoint": 0.2,
        "best_valid_target_rmsd_a": 2.5,
        "strict_valid_rate": 1.0,
        "measured_gpu_hours": 0.25,
    }


def test_aggregate_keeps_conditions_and_structure_pairs_distinct(tmp_path):
    run_dirs = []
    for condition in ("blind_guided", "blind_guided_ss"):
        for pair in (("A", "B"), ("C", "D")):
            run_dir = tmp_path / "confrover" / condition / f"case_{pair[0]}" / "seed_00"
            run_dir.mkdir(parents=True)
            (run_dir / "alternate_summary.json").write_text(
                json.dumps(_summary(*pair)), encoding="utf-8"
            )
            run_dirs.append(run_dir)

    output = aggregate_alternate_runs(run_dirs, tmp_path / "aggregate")
    aggregate = json.loads(output.read_text(encoding="utf-8"))["aggregate"]
    runs = json.loads(output.read_text(encoding="utf-8"))["runs"]

    assert {row["condition"] for row in aggregate} == {"blind_guided", "blind_guided_ss"}
    assert all(row["n_runs"] == 2 for row in aggregate)
    assert all(row["n_test_cases"] == 2 for row in aggregate)
    assert {row["seed_id"] for row in runs} == {"seed_00"}
