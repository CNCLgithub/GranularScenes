#!/usr/bin/env python

""" Submits sbatch array for rendering stimuli """
import os
import argparse
# import pandas as pd
from slurmpy import sbatch

script = 'bash {0!s}/env.d/run.sh ' + \
        '/project/scripts/experiments/quad_tree/run.sh'

def create_tasks(args):
    tasks = []
    for att in ["ac", "un"]:
        for gran in ["fixed", "multi"]:
            for scene in [1,2,3,4,5,6]:
                tasks.append((att, gran, scene, args.chains))
    return (tasks, [], [])
    
def main():
    parser = argparse.ArgumentParser(
        description = 'Submits batch jobs for Exp1',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--chains', type = int, default = 30,
                        help = 'number of chains')
    parser.add_argument('--duration', type = int, default = 15,
                        help = 'job duration (min)')
    parser.add_argument('--reversed', action = 'store_true',
                        help = 'Infer image 2 then image 1')

    args = parser.parse_args()

    tasks, kwargs, extras = create_tasks(args)
    if args['reversed']:
        kwargs['reversed'] = True

    interpreter = '#!/bin/bash'
    slurm_out = os.path.join(os.getcwd(), 'env.d/spaths/slurm')
    resources = {
        'cpus-per-task' : '1',
        'mem-per-cpu' : '8GB',
        'time' : '{0:d}'.format(args.duration),
        'partition' : 'psych_scavenge',
        'gres' : 'gpu:1',
        'requeue' : None,
        'job-name' : 'rooms',
        'chdir' : os.getcwd(),
        'output' : f"{slurm_out}/%A_%a.out",
    }
    func = script.format(os.getcwd())
    batch = sbatch.Batch(interpreter, func, tasks,
                         kwargs, extras, resources)
    job_script = batch.job_file(chunk = len(tasks), tmp_dir = slurm_out)
    job_script = '\n'.join(job_script)
    print("Template Job:")
    print(job_script)
    batch.run(n = len(tasks), script = job_script)


if __name__ == '__main__':
    main()
