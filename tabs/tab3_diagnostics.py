from dash import html, dcc
import dash_bootstrap_components as dbc

# Function signature accepts ordered crop names for capitalization
def create_tab3_layout(crop_classes):
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Model Diagnostics & Performance", className="text-light mt-4 mb-3"),
                html.P("Detailed analysis of model stability on the 20% test set (440 samples).", className="text-secondary"),
            ], width=12)
        ]),
        
        # Grid of Key Global Metrics
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(id="metric-accuracy"), className="bg-dark border-secondary"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody(id="metric-precision"), className="bg-dark border-secondary"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody(id="metric-recall"), className="bg-dark border-secondary"), width=3),
            dbc.Col(dbc.Card(dbc.CardBody(id="metric-f1"), className="bg-dark border-secondary"), width=3),
        ], className="mb-4 mt-2"),
        
        # Main Diagnostic Section (Split 7/5 columns)
        dbc.Row([
            # Left (7): Confusion Matrix Heatmap
            dbc.Col([
                dcc.Graph(id='confusion-matrix-plot')
            ], width=7, className="border-end border-secondary pe-4"),
            
            # Right (5): Detailed Table Report
            dbc.Col([
                html.Div(id="stability-report-container")
            ], width=5, className="ps-4")
        ]),
        
        html.Hr(className="my-4 bg-secondary"),
        
        # Bottom: Global Importance Bar Chart
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='global-importance-plot')
            ], width=12)
        ], className="mb-5")
        
    ], fluid=True)