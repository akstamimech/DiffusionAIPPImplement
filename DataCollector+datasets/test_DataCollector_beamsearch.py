import DataCollector_beamsearch as collector


def test_candidate_seed_is_shared_across_beta_for_same_beam():
    seed_a = collector.make_candidate_seed(selected_map=7, rank=4, beam_id=2)
    seed_b = collector.make_candidate_seed(selected_map=7, rank=4, beam_id=2)
    seed_c = collector.make_candidate_seed(selected_map=7, rank=4, beam_id=1)

    assert seed_a == seed_b
    assert seed_a != seed_c


def test_select_next_beam_keeps_three_lowest_scores():
    candidates = [
        {"id": "bad", "beam_score": 5.0},
        {"id": "best", "beam_score": 1.0},
        {"id": "third", "beam_score": 3.0},
        {"id": "second", "beam_score": 2.0},
    ]

    next_beam = collector.select_next_beam(candidates, beam_width=3)

    assert [candidate["id"] for candidate in next_beam] == ["best", "second", "third"]


def test_assigned_work_is_split_by_mpi_rank():
    tasks = list(range(10))

    rank_0_tasks = collector.assigned_tasks_for_rank(tasks, rank=0, size=3)
    rank_1_tasks = collector.assigned_tasks_for_rank(tasks, rank=1, size=3)
    rank_2_tasks = collector.assigned_tasks_for_rank(tasks, rank=2, size=3)

    assert rank_0_tasks == [0, 3, 6, 9]
    assert rank_1_tasks == [1, 4, 7]
    assert rank_2_tasks == [2, 5, 8]
    assert sorted(rank_0_tasks + rank_1_tasks + rank_2_tasks) == tasks


def test_detects_slurm_multi_task_singleton_mpi_launch():
    env = {"SLURM_NTASKS": "9"}

    assert collector.detect_mpi_launch_issue(env, mpi_size=1)
    assert not collector.detect_mpi_launch_issue(env, mpi_size=9)
    assert not collector.detect_mpi_launch_issue({}, mpi_size=1)


if __name__ == "__main__":
    test_candidate_seed_is_shared_across_beta_for_same_beam()
    test_select_next_beam_keeps_three_lowest_scores()
    test_assigned_work_is_split_by_mpi_rank()
    test_detects_slurm_multi_task_singleton_mpi_launch()
    print("beamsearch tests passed")
