"""Helper functions for the arc-flow formulation."""

import math


def find_reachable_nodes(job_sizes: list[int], capacity: int) -> list[int]:
    """Compute all reachable bin-fill levels via BFS over job sizes.

    Args:
        job_sizes: Distinct job sizes (positive integers).
        capacity: Bin (batch) capacity.

    Returns:
        Sorted list of reachable fill levels in [0, capacity].
    """
    reachable_nodes: set[int] = {0}
    queue: list[int] = [0]
    head = 0
    while head < len(queue):
        current_node = queue[head]
        head += 1
        for size in job_sizes:
            next_node = current_node + size
            if next_node <= capacity and next_node not in reachable_nodes:
                reachable_nodes.add(next_node)
                queue.append(next_node)
    return sorted(reachable_nodes)


def calculate_bin_packing_lower_bound(
    item_sizes: list[int], bin_capacity: int
) -> int:
    """Compute a simple continuous lower bound on the number of bins.

    Args:
        item_sizes: Sizes of items to pack.
        bin_capacity: Capacity of each bin.

    Returns:
        ceil(total_size / bin_capacity), or 0 if no items.
    """
    if not item_sizes:
        return 0
    total_size = sum(item_sizes)
    return math.ceil(total_size / bin_capacity)


def calculate_ofq_threshold(
    family_job_sizes: list[int], machine_capacity: int
) -> float:
    """Compute the OFQ (non-full batch) threshold for a family.

    Args:
        family_job_sizes: Sorted distinct job sizes for one family.
        machine_capacity: Batch capacity B.

    Returns:
        Threshold value O_{f,Q}.
    """
    if not family_job_sizes:
        return 0.0
    B = machine_capacity
    max_s_j = max(family_job_sizes)
    Q = 0
    for q_candidate in range(1, B + 1):
        if max_s_j <= B / q_candidate:
            Q = q_candidate
        else:
            break
    if Q == 1:
        return B / 2
    elif Q >= 2:
        return (Q - 1) * B / Q
    return 0.0
