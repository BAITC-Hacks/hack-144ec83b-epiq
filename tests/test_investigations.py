import pandas as pd
import pytest

from money_graph.investigations import common_recipients
from money_graph.repeated_routes import repeated_routes


def test_common_paths_are_directed_bounded_and_keep_large_ids():
    base = 1_000_000_087_107_9100
    ids = [base + i for i in range(8)]
    nodes = pd.DataFrame({"gid": ids, "is_seed": [True, True, True] + [False] * 5,
                          "priority_score": [0.5] * 8})
    # Two seeds merge, one isolate, one upstream-only node, and a cycle.
    edges = pd.DataFrame([(ids[a], ids[b]) for a, b in
                          [(0, 3), (1, 3), (3, 4), (4, 5), (5, 6), (6, 4), (7, 0), (7, 1)]],
                         columns=["src", "dst"])
    result, paths = common_recipients(nodes, edges, ids[:2], 2)
    assert result.gid.tolist() == ids[3:5]
    assert paths[ids[4]][ids[0]] == [ids[0], ids[3], ids[4]]
    reverse, reverse_paths = common_recipients(nodes, edges, ids[1::-1], 2)
    pd.testing.assert_frame_equal(result, reverse)
    assert paths == reverse_paths
    assert common_recipients(nodes, edges, ids[:2], 4)[0].gid.tolist() == ids[3:7]
    assert common_recipients(nodes, edges, ids[:3], 4)[0].empty
    with pytest.raises(ValueError):
        common_recipients(nodes, edges, [ids[0], ids[0]])
    with pytest.raises(ValueError):
        common_recipients(nodes, edges, [ids[0], ids[3]])


def transactions(rows):
    return pd.DataFrame(rows, columns=["src", "dst", "date", "sum_kzt"])


def test_repeated_routes_require_independent_episodes_and_ignore_duplicate_rows():
    rows = [(1, 2, "2026-07-01", 100_000), (2, 3, "2026-07-02", 95_000),
            (1, 2, "2026-07-08", 200_000), (2, 3, "2026-07-10", 180_000)]
    summary, episodes = repeated_routes(transactions(rows + rows))
    assert summary[["src", "via", "dst", "episode_count"]].values.tolist() == [[1, 2, 3, 2]]
    assert episodes.delay_days.tolist() == [1, 2]
    assert episodes.amount_ratio.tolist() == [0.95, 0.9]
    assert episodes.incoming_kzt.tolist() == [100_000, 200_000]
    assert repeated_routes(transactions(rows[:2] * 5))[0].empty


@pytest.mark.parametrize("second_in,second_out,recipient,amount", [
    ("2026-07-02", "2026-07-03", 3, 95_000),  # date windows overlap
    ("2026-07-08", "2026-07-08", 3, 95_000),  # unknown intra-day order
    ("2026-07-08", "2026-07-07", 3, 95_000),  # reverse time
    ("2026-07-08", "2026-07-12", 3, 95_000),  # outside allowed delay
    ("2026-07-08", "2026-07-09", 4, 95_000),  # different exact route
    ("2026-07-08", "2026-07-09", 3, 50_000),  # amounts incompatible
])
def test_false_repeats_are_not_counted(second_in, second_out, recipient, amount):
    tx = transactions([(1, 2, "2026-07-01", 100_000), (2, 3, "2026-07-02", 95_000),
                       (1, 2, second_in, 100_000), (2, recipient, second_out, amount)])
    assert repeated_routes(tx)[0].empty


def test_empty_routes_keep_export_schema():
    summary, episodes = repeated_routes(transactions([]))
    assert "episode_count" in summary.columns
    assert "incoming_date" in episodes.columns
