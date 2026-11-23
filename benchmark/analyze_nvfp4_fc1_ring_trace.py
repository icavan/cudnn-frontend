# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the ready/retired edges in one rank's rolling FC1 forward replay."""

import argparse
import json
import sqlite3
from pathlib import Path


def analyze(path, device, slots, capacity, live):
    if slots < 1 or capacity < 1 or not 0 <= live <= capacity:
        raise ValueError("expected positive slot and capacity counts, and 0 <= live <= capacity")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            """SELECT k.start, k.end, s.value
               FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
               WHERE k.deviceId = ? ORDER BY k.start""",
            (device,),
        ).fetchall()
    gathers = [(a, b) for a, b, name in rows if "nvfp4_ringSourceRows" in name]
    # Each scheduled live tile issues FC1, then FC2, on the same compute stream.
    gemms = [(a, b) for a, b, name in rows if "Sm100BlockScaledPersistentDenseGemmKernel" in name]
    if len(gathers) != live or len(gemms) != 2 * live:
        raise ValueError(f"expected one replay of {live} live tiles, found {len(gathers)} gathers and {len(gemms)} GEMMs")

    slots = min(slots, capacity)
    packet = min(2 * slots, capacity)
    base, refills, stalls, stall_ns = 0, 0, 0, 0
    previous = {}
    for width in dict.fromkeys((packet, slots, 1)):
        while base + width <= live:
            for i in range(width):
                slot, logical = i % slots, base + i
                start, end = gathers[logical]
                if end > gemms[2 * logical][0]:
                    raise ValueError(f"FC1 read before publication of tile {logical} in slot {slot}")
                if slot in previous:
                    retired = gemms[2 * previous[slot]][1]
                    if retired > start:
                        raise ValueError(f"slot {slot} overwritten before FC1 for tile {previous[slot]} completed")
                    if i >= slots:
                        # Portion of the idle gap after the preceding gather for which reuse
                        # was still impossible. Instrumentation changes scheduling and cost.
                        wait = max(0, retired - gathers[logical - 1][1])
                        stall_ns += wait
                        stalls += wait > 0
                        refills += 1
                previous[slot] = logical
            base += width
    result = dict(
        live_tiles=live, gathers=len(gathers), gemms=len(gemms), within_packet_refills=refills, refills_waiting=stalls, retirement_wait_us=stall_ns / 1000
    )
    for stage, intervals in (("fc1", gemms[::2]), ("fc2", gemms[1::2])):
        intersections = [sum(max(0, min(b, y) - max(a, x)) for x, y in intervals) for a, b in gathers]
        result[stage] = dict(concurrent_gathers=sum(t > 0 for t in intersections), overlap_us=sum(intersections) / 1000)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="nsys export -t sqlite output for a single profiled rolling replay")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--capacity-blocks", type=int, required=True)
    parser.add_argument("--live-blocks", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.sqlite.resolve(), args.device, args.slots, args.capacity_blocks, args.live_blocks), indent=2))
