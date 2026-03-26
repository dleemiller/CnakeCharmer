"""PageRank computation on a deterministic graph.

Keywords: graph, pagerank, ranking, iteration, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark


@python_benchmark(args=(50000,))
def pagerank(n: int) -> float:
    """Compute PageRank on n nodes for 20 iterations.

    Graph edges: i -> (i*3+1)%n, i -> (i*7+2)%n.
    Damping factor = 0.85. Each node has out-degree 2.
    Returns sum of top-10 PageRank values.

    Args:
        n: Number of nodes.

    Returns:
        Sum of top-10 PageRank values.
    """
    damping = 0.85
    iterations = 20
    out_degree = 2

    # Build adjacency: each node i has two outgoing edges
    adj0 = [0] * n
    adj1 = [0] * n
    for i in range(n):
        adj0[i] = (i * 3 + 1) % n
        adj1[i] = (i * 7 + 2) % n

    rank = [1.0 / n] * n
    new_rank = [0.0] * n

    for _ in range(iterations):
        for i in range(n):
            new_rank[i] = (1.0 - damping) / n

        for i in range(n):
            contrib = damping * rank[i] / out_degree
            new_rank[adj0[i]] += contrib
            new_rank[adj1[i]] += contrib

        rank, new_rank = new_rank, rank

    # Find top 10
    rank.sort(reverse=True)
    total = 0.0
    for i in range(min(10, n)):
        total += rank[i]
    return total
