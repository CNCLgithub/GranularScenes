#!/usr/bin/env python3

import numpy as np

def voxel_integration(ds, ws):
        depth = 0.0
        mass = 1.0
        dist = 0.0
        w = 0.0

        n = len(ws)

        # Tracing begin
        for i in range(n):
            dist += ds[i]
            w = ws[i]
            depth += mass * w * dist
            mass *= (1.0 - w)
            print(f'{w=}')
            print(f'{depth=}')
            print(f'{mass=}')

        # top off with final hit
        depth += mass * n

        print(f'{depth=}')

        return 0

voxel_integration([1, 1, 1],
                  [0.7, 0.3, 0.8])
