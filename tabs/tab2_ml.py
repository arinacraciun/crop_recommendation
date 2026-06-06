from dash import html, dcc
import dash_bootstrap_components as dbc

def create_slider(id_name, label, min_val, max_val, step, default_val):
    return dbc.Row([
        # Label remains at 4 columns
        dbc.Col(html.Label(f"{label}:", className="text-light"), width=4),
        
        # Slider is reduced to 5 columns
        dbc.Col(dcc.Slider(
            id=id_name, min=min_val, max=max_val, step=step, value=default_val,
            marks={
                min_val: {'label': str(min_val), 'style': {'color': '#f8f9fa', 'fontSize': '14px'}},
                max_val: {'label': str(max_val), 'style': {'color': '#f8f9fa', 'fontSize': '14px'}}
            }
        ), width=5),
        
        # Value readout box is increased to 3 columns to fit the arrows and 3 digits comfortably
        dbc.Col(
            dbc.Input(
                id=f"{id_name}-readout", 
                type="number", 
                value=default_val, 
                readonly=False, 
                className="text-center bg-dark text-info border-secondary px-1"
            ), width=3
        )
    ], className="mb-2 align-items-center")

def create_tab2_layout(crop_classes):
    return dbc.Container([
        dbc.Row([
            # Left Column: Input Controls
            dbc.Col([
                html.H4("Field Conditions", className="text-light mb-4"),
                
                create_slider('input-n', 'Nitrogen (N)', 0, 140, 1, 50),
                create_slider('input-p', 'Phosphorus (P)', 5, 145, 1, 50),
                create_slider('input-k', 'Potassium (K)', 5, 205, 1, 50),
                create_slider('input-temp', 'Temperature (°C)', 8, 45, 0.1, 25),
                create_slider('input-hum', 'Humidity (%)', 14, 100, 0.1, 70),
                create_slider('input-ph', 'pH Level', 3.5, 9.9, 0.1, 6.5),
                create_slider('input-rain', 'Rainfall (mm)', 20, 300, 0.1, 100),
                
                html.Div(className="d-grid gap-2 mt-4", children=[
                    dbc.Button("Predict Optimal Crop", id="predict-button", color="success", n_clicks=0)
                ])
                
            # CHANGED width from 4 to 3
            ], width=3, className="border-end border-secondary pe-4"),
            
            # Right Column: Predictions & Explainability
            dbc.Col([
                html.H4("Prediction Results", className="text-light mb-4"),
                
                # Top 3 Prediction Cards (Keep this the same)
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody(id="pred-card-1"), className="bg-dark border-success"), width=4),
                    dbc.Col(dbc.Card(dbc.CardBody(id="pred-card-2"), className="bg-dark border-secondary"), width=4),
                    dbc.Col(dbc.Card(dbc.CardBody(id="pred-card-3"), className="bg-dark border-secondary"), width=4),
                ], className="mb-4"),
                
                # Visualizations (Keep this the same, they will naturally expand)
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label("Compare to:", className="text-light fw-bold me-2 mb-0"),
                            dcc.Dropdown(
                                id='radar-crop-dropdown',
                                options=[{'label': c.capitalize(), 'value': c} for c in crop_classes],
                                value=crop_classes[0],
                                clearable=False,
                                className="text-dark",
                                style={'minWidth': '150px'}
                            )
                        ], className="d-flex align-items-center mb-2 justify-content-center"),
                        
                        dcc.Graph(id='radar-chart')
                    ], width=5),
                    
                    dbc.Col(dcc.Graph(id='shap-waterfall-plot'), width=7)
                ])
                
            # CHANGED width from 8 to 9
            ], width=9, className="ps-4")
        ], className="mt-4 mb-5")
    ], fluid=True)