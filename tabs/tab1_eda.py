from dash import html, dcc
import dash_bootstrap_components as dbc

# List of numerical features in the dataset
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def create_tab1_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("Multivariate Crop Analysis", className="text-light mb-3"),
                html.P("Select axes to explore feature distributions. Use the lasso tool on the scatter plot to filter the detailed views below.", className="text-secondary"),
            ], width=12)
        ], className="mt-4"),
        
        # Controls for the Scatter Plot
        dbc.Row([
            dbc.Col([
                html.Label("Scatter Plot X-Axis:", className="text-light"),
                dcc.Dropdown(
                    id='scatter-x-axis',
                    options=[{'label': f, 'value': f} for f in FEATURES],
                    value='N',
                    clearable=False,
                    className="text-dark"
                )
            ], width=3),
            dbc.Col([
                html.Label("Scatter Plot Y-Axis:", className="text-light"),
                dcc.Dropdown(
                    id='scatter-y-axis',
                    options=[{'label': f, 'value': f} for f in FEATURES],
                    value='rainfall',
                    clearable=False,
                    className="text-dark"
                )
            ], width=3),
        ], className="mb-4"),

        # Graph A: The Main Interactive Scatter Plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='scatter-plot')
            ], width=12)
        ]),

        html.Hr(className="my-4 bg-secondary"),

        # Graph B & C: Parallel Coordinates and Box Plot
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='parallel-plot')
            ], width=7),
            dbc.Col([
                dcc.Graph(id='box-plot')
            ], width=5)
        ], className="mb-5")
    ], fluid=True)