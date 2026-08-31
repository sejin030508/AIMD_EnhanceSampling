import numpy as np

from confmh.duet.toy import FiniteStateAdapter


def test_earlier_history_changes_proposal_with_same_endpoint():
    adapter = FiniteStateAdapter()
    route_a = [0, 1, 4]
    route_b = [0, 3, 1, 4]
    assert route_a[-1] == route_b[-1]
    assert not np.allclose(adapter.exact_q(route_a), adapter.exact_q(route_b))


def test_reencoding_uses_selected_complete_history():
    adapter = FiniteStateAdapter()
    selected = adapter.prepare_history([0, 3, 1, 4])
    assert selected["history"] == (0, 3, 1, 4)
    assert selected["route_flag"] == 0

