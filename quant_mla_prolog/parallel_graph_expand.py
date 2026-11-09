import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable

import redis
from neo4j import GraphDatabase


TASK_QUEUE = "graph:expand:queue"
VISITED_NODES = "graph:expand:visited_nodes"
VISITED_DEPTH = "graph:expand:visited_depth"
ALL_NODES = "graph:expand:all_nodes"
ALL_EDGES = "graph:expand:all_edges"


REMEMBER_DEPTH_LUA = """
local old_depth = redis.call('HGET', KEYS[1], ARGV[1])
local new_depth = tonumber(ARGV[2])
if old_depth == false or new_depth < tonumber(old_depth) then
  redis.call('HSET', KEYS[1], ARGV[1], new_depth)
  redis.call('SADD', KEYS[2], ARGV[1])
  return 1
end
return 0
"""


@dataclass(frozen=True)
class Task:
    depth: int
    node_ids: list[int] | None = None
    super_node_id: int | None = None
    after_rel_id: int = -1

    def dumps(self) -> str:
        return json.dumps(
            {
                "depth": self.depth,
                "node_ids": self.node_ids,
                "super_node_id": self.super_node_id,
                "after_rel_id": self.after_rel_id,
            },
            separators=(",", ":"),
        )

    @staticmethod
    def loads(raw: bytes) -> "Task":
        data = json.loads(raw)
        return Task(
            depth=data["depth"],
            node_ids=data.get("node_ids"),
            super_node_id=data.get("super_node_id"),
            after_rel_id=data.get("after_rel_id", -1),
        )


def chunks(items: list[int], size: int) -> Iterable[list[int]]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def enqueue(redis_client: redis.Redis, task: Task) -> None:
    redis_client.rpush(TASK_QUEUE, task.dumps())


def seed(redis_client: redis.Redis, target_ids: list[int], batch_size: int) -> None:
    pipe = redis_client.pipeline()
    pipe.delete(TASK_QUEUE, VISITED_NODES, VISITED_DEPTH, ALL_NODES, ALL_EDGES)
    if target_ids:
        pipe.sadd(VISITED_NODES, *target_ids)
        pipe.sadd(ALL_NODES, *target_ids)
        pipe.hset(VISITED_DEPTH, mapping={str(node_id): 0 for node_id in target_ids})
    pipe.execute()

    for batch in chunks(target_ids, batch_size):
        enqueue(redis_client, Task(depth=0, node_ids=batch))


def expand_batch(tx, node_ids: list[int], super_degree: int):
    result = tx.run(
        """
        MATCH (n)
        WHERE id(n) IN $node_ids
        WITH n, size((n)--()) AS degree
        WHERE degree < $super_degree
        MATCH (n)-[r]-(m)
        RETURN id(r) AS rel_id, id(m) AS neighbor_id
        """,
        node_ids=node_ids,
        super_degree=super_degree,
    )
    return [(record["rel_id"], record["neighbor_id"]) for record in result]


def find_super_nodes(tx, node_ids: list[int], super_degree: int):
    result = tx.run(
        """
        MATCH (n)
        WHERE id(n) IN $node_ids
        WITH n, size((n)--()) AS degree
        WHERE degree >= $super_degree
        RETURN id(n) AS node_id
        """,
        node_ids=node_ids,
        super_degree=super_degree,
    )
    return [record["node_id"] for record in result]


def expand_super_node_page(tx, node_id: int, after_rel_id: int, page_size: int):
    result = tx.run(
        """
        MATCH (n)-[r]-(m)
        WHERE id(n) = $node_id AND id(r) > $after_rel_id
        RETURN id(r) AS rel_id, id(m) AS neighbor_id
        ORDER BY rel_id
        LIMIT $page_size
        """,
        node_id=node_id,
        after_rel_id=after_rel_id,
        page_size=page_size,
    )
    return [(record["rel_id"], record["neighbor_id"]) for record in result]


def store_edges_and_enqueue_next(
    redis_client: redis.Redis,
    pairs: list[tuple[int, int]],
    depth: int,
    max_hops: int,
    batch_size: int,
) -> None:
    if not pairs:
        return

    rel_ids = [rel_id for rel_id, _ in pairs]
    neighbor_ids = [neighbor_id for _, neighbor_id in pairs]
    new_depth = depth + 1

    pipe = redis_client.pipeline()
    pipe.sadd(ALL_EDGES, *rel_ids)
    pipe.sadd(ALL_NODES, *neighbor_ids)
    pipe.execute()

    if new_depth >= max_hops:
        return

    remember_depth = redis_client.register_script(REMEMBER_DEPTH_LUA)
    new_nodes = [
        node_id
        for node_id in neighbor_ids
        if remember_depth(keys=[VISITED_DEPTH, VISITED_NODES], args=[node_id, new_depth]) == 1
    ]
    for batch in chunks(new_nodes, batch_size):
        enqueue(redis_client, Task(depth=depth + 1, node_ids=batch))


def worker(args) -> None:
    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=False)
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))

    try:
        while True:
            raw_task = redis_client.blpop(TASK_QUEUE, timeout=args.idle_timeout)
            if raw_task is None:
                break

            task = Task.loads(raw_task[1])
            if task.depth >= args.max_hops:
                continue

            with driver.session(database=args.neo4j_database) as session:
                if task.super_node_id is not None:
                    pairs = session.execute_read(
                        expand_super_node_page,
                        task.super_node_id,
                        task.after_rel_id,
                        args.super_page_size,
                    )
                    store_edges_and_enqueue_next(
                        redis_client, pairs, task.depth, args.max_hops, args.batch_size
                    )
                    if len(pairs) == args.super_page_size:
                        enqueue(
                            redis_client,
                            Task(
                                depth=task.depth,
                                super_node_id=task.super_node_id,
                                after_rel_id=pairs[-1][0],
                            ),
                        )
                    continue

                node_ids = task.node_ids or []
                super_nodes = session.execute_read(find_super_nodes, node_ids, args.super_degree)
                for node_id in super_nodes:
                    enqueue(redis_client, Task(depth=task.depth, super_node_id=node_id))

                pairs = session.execute_read(expand_batch, node_ids, args.super_degree)
                store_edges_and_enqueue_next(
                    redis_client, pairs, task.depth, args.max_hops, args.batch_size
                )
    finally:
        driver.close()


def collect_result(redis_client: redis.Redis) -> dict[str, list[int]]:
    return {
        "all_nodes": sorted(int(value) for value in redis_client.smembers(ALL_NODES)),
        "all_edges": sorted(int(value) for value in redis_client.smembers(ALL_EDGES)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seed", "worker", "result"], required=True)
    parser.add_argument("--target-ids", default="")
    parser.add_argument("--max-hops", type=int, required=True)
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "127.0.0.1"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "4901")))
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", "password"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--super-degree", type=int, default=50_000)
    parser.add_argument("--super-page-size", type=int, default=10_000)
    parser.add_argument("--idle-timeout", type=int, default=15)
    args = parser.parse_args()

    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=False)

    if args.mode == "seed":
        target_ids = [int(item) for item in args.target_ids.split(",") if item.strip()]
        seed(redis_client, target_ids, args.batch_size)
        print(f"seeded {len(target_ids)} target ids")
        return

    if args.mode == "worker":
        worker(args)
        return

    result = collect_result(redis_client)
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
