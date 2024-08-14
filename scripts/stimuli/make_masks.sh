#!/usr/bin/env sh
D="path_block_maze_2024-08-12"
script_dir=$(realpath "$0" | xargs dirname)
for i in $(seq 1 5);
do
    bash "${script_dir}/shuffle" "env.d/spaths/datasets/${D}/render_stairs/${i}_1.png" \
        "env.d/spaths/datasets/${D}/render_stairs/mask_${i}.png"
done
