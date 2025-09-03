import os
import pickle
import sys
import torch

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go

sys.path.append('../') 

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from hydra.utils import instantiate
from nn_lib.common.funPath import GetPaths
from nn_lib.common.funLatentSpace import GetSigmaPoints
from nn_lib.models.ARVAE import ARVAE
from nn_lib.utils import format
from plotly.subplots import make_subplots



def create_marks_slider(colormap, nb_dim):
    options = []
    cpt = 0
    for key in colormap.keys() :
        options.append({"label":key, "value":cpt})
        cpt+=1
    while cpt < nb_dim :
        options.append({"label":cpt, "value":cpt})
        cpt+=1
    return options



def main(app) :
    print("Loading parameters ...")
    path_ = sys.argv[-1] # Dir of the model
    path_model = GetPaths(path_, [".ckpt"])[".ckpt"][-1]
    D_model = torch.load(path_model)
    cfg = D_model["hyper_parameters"]["config"]

    path_embedding = os.path.join(path_, "embedding.pkl")
    with open(path_embedding, 'rb') as file_lsv:
        emb = pickle.load(file_lsv)
    
    device = torch.device(cfg.model.device)
    dim_emb = cfg.model.net.lat_dims


    print(f"Instantiating model <{cfg.datamodule._target_}>")
    Datamodule_ = instantiate(cfg.datamodule)
    Datamodule_.cfg=cfg
    Datamodule_.setup()
    full_data = Datamodule_.predict_dataloader()
    L_split = np.concatenate([
        ["train" for _ in range(len(Datamodule_.train))],
        ["val"   for _ in range(len(Datamodule_.val))],
        ["test"  for _ in range(len(Datamodule_.test))],
    ])
    L_split_colormap =  np.concatenate([
        [0 for _ in range(len(Datamodule_.train))],
        [1 for _ in range(len(Datamodule_.val))],
        [2 for _ in range(len(Datamodule_.test))],
    ])

    L_input_origin = []
    L_input_data  = []
    L_pat_name = []
    L_seq_name = []
    for sample in full_data :
        L_input_origin.append(sample["input_norm"])
        L_input_data.append(sample["input"])
        L_pat_name.append(sample["pat_name"])
        L_seq_name.append(sample["seq_name"])
    L_pat_name = np.concatenate(L_pat_name, axis=0)
    L_input_origin = np.concatenate(L_input_origin, axis=0)
    L_input_data  = np.concatenate(L_input_data, axis=0)
    L_seq_name = np.concatenate(L_seq_name, axis=0)


    if "processing" in cfg : 
        if "transform" in cfg.processing :
            transform = instantiate(cfg.processing.transform)
        else: transform = None 
        with open(cfg.processing.path_metrics, 'rb') as fileMetrics:
            D_metrics = pickle.load(fileMetrics)
    else: D_metrics={}; transform = None  

    colormap = {}
    idx_copy = -1
    for pat_name in L_pat_name :
        idx = D_metrics["pat_name"].index(pat_name)
        if idx != idx_copy : cpt = 0; idx_copy = idx

        if "keys_cond_data" in cfg.model.net.keys() : metric_keys = cfg.model.net.keys_cond_data
        else : metric_keys = D_metrics.keys()
        for key in metric_keys :
            if key in [
                "pat_name",
                "index_slices",
                "infarct_location",
                "inversion_time",
                "month",
                "sequence_type",
                "age",
                "sex",
                "thrombus",
            ] : 
                continue
            elif key in [
                "z_vals",
                "transmurality",
                "endo_surface_length",
                "infarct_size_2D",
                "angle_junction",
            ] : 
                if key not in colormap : colormap[key] = []
                colormap[key].append(D_metrics[key][idx][cpt])
            else:
                if key not in colormap : colormap[key] = []
                colormap[key].append(D_metrics[key][idx])
        cpt+=1
    for key, val in colormap.items() : colormap[key] = np.array(val)
    L_options = create_marks_slider(colormap, nb_dim=dim_emb)
    colormap["Splitting"] = L_split_colormap


    print(f"Instantiating model <ARVAE>")
    lat_dims = cfg.model.net.lat_dims

    enc = instantiate(cfg.architecture.encoders).to(device)
    dec = instantiate(cfg.architecture.decoders).to(device)

    ARVAE_model = ARVAE(
        config=cfg,
        encoder=enc,
        decoder=dec,
        lat_dim=lat_dims,
        learning_rate=cfg.model.net.lr,
        alpha=cfg.model.net.alpha,
        beta=cfg.model.net.beta,
        gamma=cfg.model.net.gamma,
        delta=cfg.model.net.delta,
    ).to(device)

    ARVAE_model.load_state_dict(
        D_model["state_dict"], 
    )



    app.layout = dbc.Container([
        # Parameters (dropdown)
        dbc.Row([
            dbc.Col(
                html.Div([
                    "2D or 3D Visualisation :", 
                    dcc.Dropdown(
                        ['2D plot', '3D plot'],
                        '2D plot',
                        id='dim_visu_plot--dropdown',
                    ),
                ]),
                width=2,
            ),
            dbc.Col(
                html.Div([
                    "Color of the points :", 
                    dcc.Dropdown(
                        list(colormap.keys()),
                        list(colormap.keys())[0],
                        id='color_pts--dropdown',
                    ),
                ]),
                width=2,
            ),
            dbc.Col(
                html.Div([
                    "cmap imgs :", 
                    dcc.Dropdown(
                        ["portland", "gray"],
                        "gray",
                        id='cmap--dropdown',
                        clearable=False,
                    ),
                ]),
                width=2,
            ),
            # dbc.Col(
            #     html.Div([
            #         "Toogle Variability in dimensions :", 
            #         dcc.RadioItems(
            #             ["yes", "no"],
            #             "no",
            #             id='bool_varia_dim--radioitem',
            #         ),
            #     ]),
            #     width=2,
            # ),
        ]),
        
        # Latent space + img recons + Dimension variabilities (Graphs)
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id='latent_space',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'New Plot'}}
                ),
                width=6,
            ),
            dbc.Col([
                dcc.Graph(
                    id='Recons_img',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'New Plot'}}
                ),
                dcc.Graph(
                    id='Dims_var',
                    config={'toImageButtonOptions': {'format': 'svg', 'filename': 'New Plot'}}
                ),
            ],
                width=6,
            )
        ]),

        # Dimensions (Sliders)
        dbc.Row([
            dbc.Col([
                html.Div([
                    "Dimension 1 :", 
                    dcc.Dropdown(
                        options=L_options,
                        value=0,
                        id='dim1--slider',
                        clearable=False,
                    ),
                ]),
            ], width=2
            ),
            dbc.Col([
                html.Div([
                    "Dimension 2 :", 
                    dcc.Dropdown(
                        options=L_options,
                        value=1,
                        id='dim2--slider',
                        clearable=False,
                    ),
                ]),
            ], width=2
            ),
            dbc.Col([
                html.Div([
                    "Dimension 3 :", 
                    dcc.Dropdown(
                        options=L_options,
                        value=2,
                        id='dim3--slider',
                        clearable=False,
                    ),
                ]),
            ], width=2
            ),
            dbc.Col(
                html.Div([
                    "Dimension variabilities :", 
                    dcc.Slider(
                        0,
                        np.shape(emb)[-1]-1,
                        step=1,
                        id='dim_var--slider',
                        value=0,
                    ),
                ]),
                width=6,
            )
        ]),

    ], fluid=True)



    # Callback for Latent Space + Sliders
    @app.callback(
        Output('latent_space', 'figure'),
        Input('dim1--slider', 'value'),
        Input('dim2--slider', 'value'),
        Input('dim3--slider', 'value'),
        Input('dim_visu_plot--dropdown', 'value'),
        Input('color_pts--dropdown', 'value'),
    )
    def _func(dim1, dim2, dim3, dim_plot, color_item):
        graph_layout = go.Layout(
            title="Latent space: dim "+str(dim1)+"-"+str(dim2)+"-"+str(dim3),
            height=900,
            # margin=dict(l=10, r=10, b=10, t=50),
            autosize=True,
            uirevision=True,
            template="plotly_white"
        )
        marker_dict = dict(
            size=8,
            color=colormap[color_item],
            cmin=colormap[color_item].min(),
            cmax=colormap[color_item].max(),
            colorbar=dict(title="Colorbar"),
            colorscale="portland",
        )

        fig = go.Figure(layout=graph_layout)
        if dim_plot == "2D plot" :
            plot_2D = go.Scatter(
                x = emb[:,dim1],
                y = emb[:,dim2],
                mode='markers',
                text=L_pat_name,
                customdata=np.dstack((
                    L_seq_name,
                    colormap[color_item],
                    L_split,
                    )
                ).squeeze(),
                hovertemplate='<br>'.join([
                    'x= %{x}',
                    'y= %{y}',
                    'pat_name= %{text}',
                    'seq_name= %{customdata[0]}',
                    'colormap= %{customdata[1]}',
                    'spliting= %{customdata[2]}',
                ]),
                marker=marker_dict,
            )
            fig.add_trace(plot_2D)
            fig.update_layout(
                xaxis_title="dim "+str(dim1),
                yaxis_title="dim "+str(dim2),
            )

        if dim_plot == "3D plot" :
            plot_3D = go.Scatter3d(
                x = emb[:,dim1],
                y = emb[:,dim2],
                z = emb[:,dim3],
                mode='markers',
                text=L_pat_name,
                customdata=np.dstack((
                    L_seq_name,
                    colormap[color_item],
                    L_split,
                    )
                ).squeeze(),
                hovertemplate='<br>'.join([
                    'x= %{x}',
                    'y= %{y}',
                    'pat_name= %{text}',
                    'seq_name= %{customdata[0]}',
                    'colormap= %{customdata[1]}',
                    'spliting= %{customdata[2]}',
                ]),
                marker=marker_dict,
            )
            fig.add_trace(plot_3D)
            fig.update_layout(
                scene = dict(
                    xaxis_title="dim "+str(dim1),
                    yaxis_title="dim "+str(dim2),
                    zaxis_title="dim "+str(dim3),
                ),
            )
        return fig


    # Callback for Recons imgs
    @app.callback(
        Output('Recons_img', 'figure'),
        Input('latent_space', 'clickData'),
        Input('cmap--dropdown', 'value'),
    )
    def _func(clickData, cmap):
        graph_layout = go.Layout(
            height=500,
            # margin=dict(l=10, r=10, b=10, t=50),
            xaxis={'anchor': 'y', 'constrain': 'domain', 'domain': [0.0, 1/3], 'scaleanchor': 'y'},
            yaxis={'anchor': 'x', 'autorange': 'reversed', 'constrain': 'domain', 'domain': [0.0, 1.0]},
            autosize=True,
            uirevision=True,
            template="plotly_white",
        )
        fig = make_subplots(1, 3,
            subplot_titles=["Original Image", "Reconstructed Image", "Errors"]
        )
        fig.update_layout(graph_layout)

        if clickData :
            pt_idx = clickData["points"][0]["pointNumber"]
            img = L_input_origin[pt_idx]

            vec_emb = np.expand_dims(emb[pt_idx,:], axis=0)
            vec_emb = torch.tensor(vec_emb).to(device)
            recons_img = ARVAE_model.decoder(vec_emb)
            recons_img = format.torch_to_numpy(recons_img).squeeze()
            if "segmentation" in cfg.architecture.decoders.keys() :
                recons_img = np.argmax(recons_img, axis=0)
                zmax = 2
            else: zmax = 1
            if transform : recons_img = transform.wrap(recons_img)

            errors_img = abs(img - recons_img)
            val_error = np.nansum(errors_img)
            
            for i,data_plot in enumerate([img, recons_img, errors_img]) :
                if i == 0 : showscale=True
                else: showscale=False
                plot_ = go.Heatmap(
                    z = data_plot,
                    colorscale=cmap,
                    showscale=showscale,
                    yaxis="y"+str(i+1),
                    zmin=0,
                    zmax=zmax,
                )        
                fig.add_trace(plot_, 1,i+1)
                fig.update_layout({
                    'xaxis'+str(i+1): {'anchor': 'y'+str(i+1), 'domain': [i/3, (i+1)/3], 'matches': 'x', 'visible': False},
                    'yaxis'+str(i+1): {'anchor': 'x'+str(i+1), 'domain': [0.0, 1.0], 'matches': 'y', 'showticklabels': False, 'visible': False},
                })
            fig.update_layout(title="Image & Reconstruction, Error (%) = "+str((val_error/recons_img.size)*100)[:5])
        return fig


    # Callback for Dimension Variability
    @app.callback(
        Output('Dims_var', 'figure'),
        Input('dim_var--slider', 'value'),
        Input('cmap--dropdown', 'value'),
    )
    def _func(dim_var, cmap) :
        graph_layout = go.Layout(
            title="Variations in dimension "+str(dim_var),
            height=365,
            # margin=dict(l=10, r=10, b=10, t=50),
            xaxis={'anchor': 'y', 'constrain': 'domain', 'domain': [0.0, 1/5], 'scaleanchor': 'y'},
            yaxis={'anchor': 'x', 'autorange': 'reversed', 'constrain': 'domain', 'domain': [0.0, 1.0]},
            autosize=True,
            uirevision=True,
            template="plotly_white",
        )
        fig = make_subplots(1, 5,
            subplot_titles=[str(i)+" sigma" for i in range (-2,3,1)],
        )
        fig.update_layout(graph_layout)

        sigma_pts = GetSigmaPoints(emb, np.shape(emb)[-1], d1=dim_var, d2=0)
        sigma_pts = torch.tensor(sigma_pts).to(device)

        inv_sigma_pts = np.squeeze(ARVAE_model.decoder(sigma_pts))
        inv_sigma_pts = format.torch_to_numpy(inv_sigma_pts)
        if "segmentation" in cfg.architecture.decoders.keys() :
            inv_sigma_pts = np.argmax(inv_sigma_pts, axis=1)
            zmax = 2
        else: zmax = 1
        if transform: inv_sigma_pts = transform.wrap(inv_sigma_pts)


        for i,data_plot in enumerate(inv_sigma_pts[:5]) :
            if i == 0 : showscale=True
            else: showscale=False
            plot_ = go.Heatmap(
                z = data_plot,
                colorscale=cmap,
                showscale=showscale,
                yaxis="y"+str(i+1),
                zmin=0,
                zmax=zmax,
            )        
            fig.add_trace(plot_, 1,i+1)
            fig.update_layout({
                'xaxis'+str(i+1): {'anchor': 'y'+str(i+1), 'domain': [i/5, (i+1)/5], 'matches': 'x', 'visible': False},
                'yaxis'+str(i+1): {'anchor': 'x'+str(i+1), 'domain': [0.0, 1.0], 'matches': 'y', 'showticklabels': False, 'visible': False},
            })
        return fig



if __name__ == "__main__" :
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    main(app)    
    app.run_server(debug=True)
