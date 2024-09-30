#!/usr/bin/env python3
import os
import csv
import json
import argparse

def main():

    parser = argparse.ArgumentParser(
        description = 'Generates condlist for experiment',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type = str,
                        help = "Which scene dataset to use",
                        default = 'path_block_2024-03-14')
    args = parser.parse_args()

    dataset = '/spaths/datasets/' + args.dataset


    aa = []
    ab = []
    with open(dataset + '/scenes.csv', 'r') as f:
        reader = csv.reader(f)
        # groupby move
        for (i, row) in enumerate(reader):
            if i == 0:
                continue
            print(row)
            scene = int(row[0])
            flipx = row[1] == 'true'
            for door in [1, 2]:
                # first create each `a->a` trial
                a = '{0:d}_{1:d}.png'.format(scene, door)
                aa.append([a, a, flipx])
                # then proceed to make `a -> b` trials
                b = '{0:d}_{1:d}_blocked.png'.format(scene, door)
                ab.append([a, b, flipx])

    # repeate aa trials to have a 50/50 split
    naa = len(aa)
    nab = len(ab)

    trials = [aa + ab]
    with open(os.path.join(dataset, 'condlist.json'), 'w') as f:
       json.dump(trials, f)


if __name__ == '__main__':
    main()
