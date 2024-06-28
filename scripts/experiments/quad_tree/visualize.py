#!/usr/bin/env python3

import numpy as np
# import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

def downsample(a, n:int = 2):
    if n ==1 :
        return a
    b = a.shape[0]//n
    a_downsampled = a.reshape(-1, n, b, n).sum((-1, -3)) / (n*n)
    return a_downsampled

# EXPNAME = 'ccn_2023_exp'
EXPNAME = 'path_block_2024-03-14'
burn_in = 1
scale = 8
SUBPLOT_WIDTH=300

def main():
    scenes = [1, 2, 3]
    titles = ['geo', 'att', 'pmat']


    # df_path = f"/spaths/datasets/{EXPNAME}/scenes.csv"
    # df = pd.read_csv(df_path)
    # df = df.loc[map(lambda x: x in scenes, df['id'])]

    row_count = 1
    geo_fig = make_subplots(rows=len(scenes), cols=2,
                        shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles = ["Left", "Right"])

    att_fig = make_subplots(rows=len(scenes), cols=2,
                            shared_xaxes=True,
                            shared_yaxes=True,
                            subplot_titles = ["Left", "Right"])


    pmat_fig = make_subplots(rows=len(scenes), cols=2,
                             shared_xaxes=True,
                             shared_yaxes=True,
                             subplot_titles = ["Left", "Right"])

    exp_path = f'/spaths/experiments/{EXPNAME}'

    for scene in [1,2,3]:
        for door in [1,2]:
            data_path = f'{exp_path}/{scene}_{door}_ac_aggregated.npz'
            data = np.load(data_path)

            geo = np.mean(data['geo'], axis = (0,1))
            geo_hm =  go.Heatmap(z = geo.T, coloraxis="coloraxis1")
            geo_fig.update_yaxes(title_text=f"Scene: {scene}", row=scene, col=1)
            geo_fig.add_trace(geo_hm, row = scene, col = door)


            att = np.mean(np.clip(data['att'], -20., 0.), axis = (0,1))
            att_hm =  go.Heatmap(z = att.T, coloraxis="coloraxis1")
            att_fig.update_yaxes(title_text=f"Scene: {scene}", row=scene, col=1)
            att_fig.add_trace(att_hm, row = scene, col = door)

            pmat = np.mean(data['pmat'], axis = (0,1))
            pmat_hm =  go.Heatmap(z = pmat.T, coloraxis="coloraxis1")
            pmat_fig.update_yaxes(title_text=f"Scene: {scene}", row=scene, col=1)
            pmat_fig.add_trace(pmat_hm, row = scene, col = door)

            img = np.mean(data['img'], axis = (0, 3))
            img = np.flip(img, 0)
            img = np.rot90(img)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(f'{exp_path}/ac_{scene}_{door}_img.png')


    geo_fig.update_layout(
        height = SUBPLOT_WIDTH * 3,
        width = SUBPLOT_WIDTH * 2 + 15,
        coloraxis1=dict(colorscale='blues'),
        showlegend=False
    )
    geo_fig.write_html(f'{exp_path}/ac_geo.html')
    geo_fig.write_image(f'{exp_path}/ac_geo.png')


    att_fig.update_layout(
        height = SUBPLOT_WIDTH * 3,
        width = SUBPLOT_WIDTH * 2 + 15,
        coloraxis1=dict(colorscale='reds'),
        showlegend=False
    )
    att_fig.write_html(f'{exp_path}/ac_att.html')
    att_fig.write_image(f'{exp_path}/ac_att.png')

    pmat_fig.update_layout(
        height = SUBPLOT_WIDTH * 3,
        width = SUBPLOT_WIDTH * 2 + 15,
        coloraxis1=dict(colorscale='greens'),
        showlegend=False
    )
    pmat_fig.write_html(f'{exp_path}/ac_path.html')
    pmat_fig.write_image(f'{exp_path}/ac_path.png')

if __name__ == '__main__':
    main()
