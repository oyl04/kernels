import argparse
import concurrent.futures
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any

import redis


DEFAULT_GRAPH = "parallel_speedup_bench"


@dataclass(frozen=True)
class QueryResult:
    nodes: set[int]
    edges: set[int]
    seconds: float


def redis_client(args) -> redis.Redis:
    return redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)


def graph_query(client: redis.Redis, graph: str, query: str) -> Any:
    return client.execute_command("GRAPH.QUERY", graph, query, "--compact")


def row_values(result: Any) -> list[list[Any]]:
    if not result or len(result) < 2:
        return []
    rows = result[1]
    if not isinstance(rows, list):
        return []
    return [[decode_cell(cell) for cell in row] for row in rows]


def scalar_rows(result: Any) -> list[Any]:
    return [row[0] for row in row_values(result)]


def decode_cell(value: Any) -> Any:
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], int)
        and 0 <= value[0] <= 10
    ):
        return decode_cell(value[1])
    if isinstance(value, list):
        return [decode_cell(item) for item in value]
    return value


def create_graph(args) -> list[int]:
    client = redis_client(args)
    try:
        client.execute_command("GRAPH.DELETE", args.graph)
    except redis.ResponseError:
        pass

    graph_query(
        client,
        args.graph,
        f"""
        UNWIND range(0, {args.targets - 1}) AS c
        CREATE (:Target {{component: c}})
        """,
    )

    if args.shared_super_targets:
        graph_query(
            client,
            args.graph,
            f"""
            CREATE (h:Hub {{component: -1, super: true, shared: true}})
            WITH h
            UNWIND range(0, {args.super_fanout - 1}) AS i
            CREATE (h)-[:LINK]->(:Leaf {{component: -1, idx: i}})
            """,
        )
        graph_query(
            client,
            args.graph,
            f"""
            MATCH (h:Hub {{shared: true}})
            MATCH (t:Target)
            WHERE t.component < {args.shared_super_targets}
            CREATE (t)-[:TO]->(h)
            """,
        )

    for component in range(args.targets):
        if component < args.super_components:
            graph_query(
                client,
                args.graph,
                f"""
                MATCH (t:Target {{component: {component}}})
                CREATE (t)-[:TO]->(h:Hub {{component: {component}, super: true}})
                WITH h
                UNWIND range(0, {args.super_fanout - 1}) AS i
                CREATE (h)-[:LINK]->(:Leaf {{component: {component}, idx: i}})
                """,
            )
        graph_query(
            client,
            args.graph,
            f"""
            MATCH (t:Target {{component: {component}}})
            UNWIND range(0, {args.normal_fanout - 1}) AS i
            CREATE (t)-[:TO]->(:Ord {{component: {component}, idx: i}})
            """,
        )
        graph_query(
            client,
            args.graph,
            f"""
            MATCH (t:Target {{component: {component}}})
            CREATE (t)-[:TO]->(:Chain {{component: {component}, depth: 1}})
                  -[:TO]->(:Chain {{component: {component}, depth: 2}})
                  -[:TO]->(:Chain {{component: {component}, depth: 3}})
            """,
        )

    result = graph_query(
        client,
        args.graph,
        "MATCH (t:Target) RETURN id(t) ORDER BY t.component",
    )
    target_ids = [int(value) for value in scalar_rows(result)]
    print(json.dumps({"graph": args.graph, "target_ids": target_ids}))
    return target_ids


def query_for_targets(client: redis.Redis, args, target_ids: list[int]) -> QueryResult:
    ids = ",".join(str(node_id) for node_id in target_ids)
    started = time.perf_counter()
    result = graph_query(
        client,
        args.graph,
        f"""
        MATCH p=(a)-[*0..{args.max_hops}]-(b)
        WHERE id(a) IN [{ids}]
        WITH nodes(p) AS ns, relationships(p) AS rs
        UNWIND ns AS n
        UNWIND rs AS r
        RETURN collect(DISTINCT id(n)) AS all_nodes,
               collect(DISTINCT id(r)) AS all_edges
        """,
    )
    elapsed = time.perf_counter() - started
    rows = row_values(result)
    if not rows:
        return QueryResult(set(), set(), elapsed)
    nodes, edges = rows[0]
    return QueryResult({int(v) for v in nodes}, {int(v) for v in edges}, elapsed)


def find_super_nodes(client: redis.Redis, args, node_ids: list[int]) -> set[int]:
    if not node_ids:
        return set()
    ids = ",".join(str(node_id) for node_id in node_ids)
    result = graph_query(
        client,
        args.graph,
        f"""
        MATCH (n)
        WHERE id(n) IN [{ids}] AND n.super = true
        RETURN id(n)
        """,
    )
    return {int(value) for value in scalar_rows(result)}


def expand_regular_batch(client: redis.Redis, args, node_ids: list[int]) -> tuple[set[int], set[int]]:
    if not node_ids:
        return set(), set()
    ids = ",".join(str(node_id) for node_id in node_ids)
    result = graph_query(
        client,
        args.graph,
        f"""
        MATCH (n)-[r]-(m)
        WHERE id(n) IN [{ids}]
        RETURN collect(DISTINCT id(m)) AS nodes,
               collect(DISTINCT id(r)) AS edges
        """,
    )
    rows = row_values(result)
    if not rows:
        return set(), set()
    nodes, edges = rows[0]
    return {int(v) for v in nodes}, {int(v) for v in edges}


def super_edge_bounds(client: redis.Redis, args, node_id: int) -> tuple[int, int] | None:
    result = graph_query(
        client,
        args.graph,
        f"""
        MATCH (n)-[r]-(m)
        WHERE id(n) = {node_id}
        RETURN min(id(r)), max(id(r))
        """,
    )
    rows = row_values(result)
    if not rows or rows[0][0] is None or rows[0][1] is None:
        return None
    return int(rows[0][0]), int(rows[0][1])


def expand_super_page(
    client: redis.Redis,
    args,
    node_id: int,
    rel_id_start: int,
    rel_id_end: int,
) -> tuple[set[int], set[int]]:
    result = graph_query(
        client,
        args.graph,
        f"""
        MATCH (n)-[r]-(m)
        WHERE id(n) = {node_id}
          AND id(r) >= {rel_id_start}
          AND id(r) <= {rel_id_end}
        RETURN id(m) AS node_id, id(r) AS edge_id
        """,
    )
    rows = row_values(result)
    return {int(row[0]) for row in rows}, {int(row[1]) for row in rows}


def adaptive_parallel(args, target_ids: list[int]) -> QueryResult:
    started = time.perf_counter()
    all_nodes = set(target_ids)
    all_edges: set[int] = set()
    visited = set(target_ids)
    frontier = set(target_ids)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for _depth in range(args.max_hops):
            if not frontier:
                break

            client = redis_client(args)
            super_nodes = find_super_nodes(client, args, sorted(frontier))
            regular_nodes = sorted(frontier - super_nodes)

            futures: list[concurrent.futures.Future[tuple[set[int], set[int]]]] = []
            for offset in range(0, len(regular_nodes), args.batch_size):
                batch = regular_nodes[offset : offset + args.batch_size]
                futures.append(
                    executor.submit(expand_regular_batch, redis_client(args), args, batch)
                )

            for node_id in sorted(super_nodes):
                bounds = super_edge_bounds(client, args, node_id)
                if bounds is None:
                    continue
                min_rel_id, max_rel_id = bounds
                for rel_id_start in range(min_rel_id, max_rel_id + 1, args.super_page_size):
                    rel_id_end = min(rel_id_start + args.super_page_size - 1, max_rel_id)
                    futures.append(
                        executor.submit(
                            expand_super_page,
                            redis_client(args),
                            args,
                            node_id,
                            rel_id_start,
                            rel_id_end,
                        )
                    )

            next_frontier: set[int] = set()
            for future in concurrent.futures.as_completed(futures):
                nodes, edges = future.result()
                all_nodes.update(nodes)
                all_edges.update(edges)
                next_frontier.update(nodes - visited)

            visited.update(next_frontier)
            frontier = next_frontier

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "mode": "adaptive_parallel",
                "workers": args.workers,
                "seconds": round(elapsed, 4),
                "nodes": len(all_nodes),
                "edges": len(all_edges),
            }
        )
    )
    return QueryResult(all_nodes, all_edges, elapsed)


def sequential(args, target_ids: list[int]) -> QueryResult:
    client = redis_client(args)
    all_nodes: set[int] = set()
    all_edges: set[int] = set()
    started = time.perf_counter()
    per_query_seconds: list[float] = []

    for target_id in target_ids:
        result = query_for_targets(client, args, [target_id])
        all_nodes.update(result.nodes)
        all_edges.update(result.edges)
        per_query_seconds.append(result.seconds)

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "mode": "sequential",
                "seconds": round(elapsed, 4),
                "per_query_median": round(statistics.median(per_query_seconds), 4),
                "nodes": len(all_nodes),
                "edges": len(all_edges),
            }
        )
    )
    return QueryResult(all_nodes, all_edges, elapsed)


def parallel(args, target_ids: list[int]) -> QueryResult:
    all_nodes: set[int] = set()
    all_edges: set[int] = set()
    started = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(query_for_targets, redis_client(args), args, [target_id])
            for target_id in target_ids
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            all_nodes.update(result.nodes)
            all_edges.update(result.edges)

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "mode": "parallel",
                "workers": args.workers,
                "seconds": round(elapsed, 4),
                "nodes": len(all_nodes),
                "edges": len(all_edges),
            }
        )
    )
    return QueryResult(all_nodes, all_edges, elapsed)


def benchmark(args) -> None:
    target_ids = create_graph(args)
    target_ids = target_ids[: args.targets]

    seq = sequential(args, target_ids)
    par = parallel(args, target_ids)
    adaptive = adaptive_parallel(args, target_ids)

    if seq.nodes != par.nodes or seq.edges != par.edges:
        raise SystemExit(
            json.dumps(
                {
                    "ok": False,
                    "reason": "parallel result differs from sequential result",
                    "sequential_nodes": len(seq.nodes),
                    "parallel_nodes": len(par.nodes),
                    "sequential_edges": len(seq.edges),
                    "parallel_edges": len(par.edges),
                }
            )
        )
    if seq.nodes != adaptive.nodes or seq.edges != adaptive.edges:
        raise SystemExit(
            json.dumps(
                {
                    "ok": False,
                    "reason": "adaptive result differs from sequential result",
                    "sequential_nodes": len(seq.nodes),
                    "adaptive_nodes": len(adaptive.nodes),
                    "sequential_edges": len(seq.edges),
                    "adaptive_edges": len(adaptive.edges),
                }
            )
        )

    speedup = seq.seconds / par.seconds if par.seconds else float("inf")
    adaptive_speedup = seq.seconds / adaptive.seconds if adaptive.seconds else float("inf")
    report = {
        "ok": max(speedup, adaptive_speedup) >= args.min_speedup,
        "target_parallel_speedup": round(speedup, 3),
        "adaptive_parallel_speedup": round(adaptive_speedup, 3),
        "required_speedup": args.min_speedup,
        "sequential_seconds": round(seq.seconds, 4),
        "target_parallel_seconds": round(par.seconds, 4),
        "adaptive_parallel_seconds": round(adaptive.seconds, 4),
        "target_count": len(target_ids),
        "super_fanout": args.super_fanout,
        "normal_fanout": args.normal_fanout,
        "max_hops": args.max_hops,
        "super_components": args.super_components,
        "shared_super_targets": args.shared_super_targets,
    }
    print(json.dumps(report))

    if max(speedup, adaptive_speedup) < args.min_speedup:
        raise SystemExit(
            "Speedup target was not reached. Increase --super-fanout, increase --targets, "
            "or set --workers close to FalkorDB's query thread capacity."
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default=DEFAULT_GRAPH)
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=4901)
    parser.add_argument("--targets", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--super-fanout", type=int, default=2500)
    parser.add_argument("--super-components", type=int, default=0)
    parser.add_argument("--shared-super-targets", type=int, default=8)
    parser.add_argument("--super-page-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--normal-fanout", type=int, default=80)
    parser.add_argument("--min-speedup", type=float, default=1.25)
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
