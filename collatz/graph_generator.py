#!/usr/bin/env python3
from collections import defaultdict

sequences = dict()


def get_collatz_memo(k):
    sequence = []

    while k != 1:
        if k in sequences:
            for i in range(len(sequence)):
                sequences[sequence[i]] = sequence[i:] + sequences[k]
            return sequence + sequences[k]

        sequence.append(k)
        if k % 2 == 0:
            k //= 2
        else:
            k = k * 3 + 1

    sequence.append(k)

    for i in range(len(sequence)):
        if sequence[i] in sequences:
            break
        sequences[sequence[i]] = sequence[i:]

    return sequence


def makeDot(paths, filepath):
    edges = set()
    dists = defaultdict(set)
    for path in paths.values():
        for i in range(len(path) - 1):
            edge = tuple(path[i : i + 2])
            edges.add(tuple(path[i : i + 2]))
            dists[len(path) - i].add(edge[0])
    nodeAttributes = "\n".join(
        [
            f"\t{k} [shape=circle; style=filled; fillcolor={len(v) % 10 + 1}];"
            for k, v in paths.items()
        ]
    )
    edgeText = "\n".join([f"\t{x[0]}->{x[1]}" for x in edges])
    rankNodes = {k: "; ".join([str(x) for x in v]) for k, v in dists.items()}
    rankText = "\n".join([f"\t{{rank = same; {x};}}" for x in rankNodes.values()])
    dot = f"""
digraph G {{
rankdir = RL;
     subgraph {{
     node [colorscheme=spectral10]

{nodeAttributes}

{edgeText}

{rankText}
     }}
}}"""
    with open(filepath, "w") as out:
        out.write(dot)

if __name__ == "__main__":

    for i in range(1, 1001):
        sequences[i] = get_collatz_memo(i)

    makeDot({i:sequences[i] for i in range(1, 21)}, "graph_20.dot")
    makeDot({i:sequences[i] for i in range(1, 31)}, "graph_30.dot")
    makeDot({i:sequences[i] for i in range(1, 1001)}, "graph_1000.dot")