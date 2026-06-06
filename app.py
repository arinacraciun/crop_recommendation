import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import shap
import numpy as np
import os
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Import tab layouts
from tabs.tab1_eda import create_tab1_layout
from tabs.tab2_ml import create_tab2_layout
from tabs.tab3_diagnostics import create_tab3_layout


# Initialize the Dash app with a dark theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Load ML Artifacts
model = joblib.load('models/lgbm_model.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')
global_importance_df = pd.read_csv('models/global_importance.csv')
crop_means_df = pd.read_csv('models/crop_means.csv')

confusion_matrix_data = joblib.load('models/test_confusion_matrix.joblib')
global_importance_df = pd.read_csv('models/global_importance.csv')
test_targets = np.load('models/test_targets.npy')
test_preds = np.load('models/test_preds.npy')

# Get ordered crop names directly from the encoder
ordered_crop_names = list(label_encoder.classes_)
ordered_crop_names_capitalized = [c.capitalize() for c in ordered_crop_names]

# Setup SHAP Explainer
explainer = shap.TreeExplainer(model)

# Load the data
# Ensure 'Crop_recommendation.csv' is inside the 'data' folder
try:
    df = pd.read_csv('data/Crop_recommendation.csv')
except FileNotFoundError:
    # Dummy dataframe fallback for testing the layout before adding the CSV
    df = pd.DataFrame({
        'N': [90, 85, 60], 'P': [42, 58, 55], 'K': [43, 41, 44],
        'temperature': [20.8, 21.7, 23.0], 'humidity': [82.0, 80.3, 82.3],
        'ph': [6.5, 7.0, 7.8], 'rainfall': [202.9, 226.6, 263.9],
        'label': ['rice', 'rice', 'maize']
    })

# Force IDs to strings to survive Dash JSON serialization perfectly
df['id'] = df.index.astype(str)

# Create a global color mapping that never changes
GLOBAL_LABEL_MAPPING = {label: idx for idx, label in enumerate(df['label'].unique())}
MAX_LABEL_IDX = len(GLOBAL_LABEL_MAPPING) - 1

FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Create a strict color map using a palette with 26 distinct colors
UNIQUE_CROPS = df['label'].unique()
CROP_COLORS = px.colors.qualitative.Alphabet[:len(UNIQUE_CROPS)]
COLOR_MAP = {crop: color for crop, color in zip(UNIQUE_CROPS, CROP_COLORS)}

# Extract sorted crop names from the means dataframe
crop_classes = sorted(crop_means_df['label'].unique())

# Application Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Agricultural Intelligence Dashboard", className="text-center text-light mt-4 mb-4"))
    ]),
    
    dbc.Tabs([
        dbc.Tab(create_tab1_layout(), label="Exploratory Analysis", tab_id="tab-1"),
        dbc.Tab(create_tab2_layout(crop_classes), label="Crop Predictor", tab_id="tab-2"),
        dbc.Tab(create_tab3_layout(ordered_crop_names_capitalized), label="Model Diagnostics", tab_id="tab-3"),
    ], id="tabs", active_tab="tab-1"),
    
], fluid=True, style={"backgroundColor": "#222222", "minHeight": "100vh"})


# -----------------------------------------------------------------------------
# Callbacks for Tab 1 Cross-Filtering
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Callback 1: Update ONLY the Scatter Plot (Triggered by Dropdowns)
# -----------------------------------------------------------------------------
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-x-axis', 'value'),
     Input('scatter-y-axis', 'value')]
)
def update_scatter(x_col, y_col):
    fig_scatter = px.scatter(
        df, x=x_col, y=y_col, 
        color='label',
        symbol='label',                 # This assigns unique shapes to each crop
        color_discrete_map=COLOR_MAP,   # This locks the colors
        custom_data=['id'], 
        title=f"{x_col.capitalize()} vs {y_col.capitalize()} (All Crops)",
        template="plotly_dark"
    )
    
    # Make markers slightly larger so the shapes are distinct
    fig_scatter.update_traces(marker=dict(size=6))
    fig_scatter.update_layout(clickmode='event+select', transition_duration=500)
    
    return fig_scatter


# -----------------------------------------------------------------------------
# Callback 2: Update Parallel & Box Plots (Triggered by Selection or Dropdowns)
# -----------------------------------------------------------------------------
@app.callback(
    [Output('parallel-plot', 'figure'),
     Output('box-plot', 'figure')],
    [Input('scatter-plot', 'selectedData'),
     Input('scatter-x-axis', 'value')]
)
def update_bottom_graphs(selected_data, x_col):
    filtered_df = df.copy()
    
    # Extract IDs safely and force string comparison
    if selected_data and selected_data.get('points'):
        selected_indices = [str(point['customdata'][0]) for point in selected_data['points'] if 'customdata' in point]
        filtered_df = df[df['id'].isin(selected_indices)]
    
    # 1. Parallel Coordinates Plot
    # Use the global mapping instead of recalculating it
    filtered_df['label_num'] = filtered_df['label'].map(GLOBAL_LABEL_MAPPING)
    
    fig_parallel = px.parallel_coordinates(
        filtered_df, 
        dimensions=FEATURES,
        color='label_num',
        title="Nutrient & Climate Profile (Selected Data)",
        template="plotly_dark",
        color_continuous_scale=px.colors.diverging.Tealrose,
        range_color=[0, MAX_LABEL_IDX]  # This prevents the division-by-zero blank graph error
    )
    fig_parallel.layout.coloraxis.showscale = False

    # 2. Box Plot
    fig_box = px.box(
        filtered_df, x='label', y=x_col, 
        color='label',
        color_discrete_map=COLOR_MAP,   # This ensures the filtered colors match the scatter plot
        title=f"Distribution of {x_col.capitalize()} (Selected Data)",
        template="plotly_dark"
    )
    fig_box.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})

    return fig_parallel, fig_box

# -----------------------------------------------------------------------------
# Callbacks for Tab 2 ML Predictor
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Callback 2A: Main Prediction & SHAP (Triggered by Predict Button)
# -----------------------------------------------------------------------------
@app.callback(
    [Output('pred-card-1', 'children'),
     Output('pred-card-2', 'children'),
     Output('pred-card-3', 'children'),
     Output('shap-waterfall-plot', 'figure'),
     Output('radar-crop-dropdown', 'value')], # This syncs the dropdown to the top prediction
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('input-n', 'value'),
     dash.dependencies.State('input-p', 'value'),
     dash.dependencies.State('input-k', 'value'),
     dash.dependencies.State('input-temp', 'value'),
     dash.dependencies.State('input-hum', 'value'),
     dash.dependencies.State('input-ph', 'value'),
     dash.dependencies.State('input-rain', 'value')]
)
def update_prediction(n_clicks, n, p, k, temp, hum, ph, rain):
    input_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_values = [n, p, k, temp, hum, ph, rain]
    input_data = pd.DataFrame([input_values], columns=input_features)
    
    # 1. Get Top 3 Predictions
    probabilities = model.predict_proba(input_data)[0]
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    top_3_probs = probabilities[top_3_indices]
    top_3_crops = label_encoder.inverse_transform(top_3_indices)
    
    # Format UI Cards
    def create_card_content(rank, crop, prob, is_top=False):
        color = "text-success" if is_top else "text-info"
        return [
            html.H6(f"#{rank} Match", className="text-secondary"),
            html.H3(crop.capitalize(), className=f"{color} fw-bold"),
            html.H4(f"{prob*100:.1f}%", className="text-light")
        ]
        
    card1 = create_card_content(1, top_3_crops[0], top_3_probs[0], is_top=True)
    card2 = create_card_content(2, top_3_crops[1], top_3_probs[1])
    card3 = create_card_content(3, top_3_crops[2], top_3_probs[2])
    
    # 2. Local SHAP Waterfall Plot
    explanation = explainer(input_data)
    class_index = top_3_indices[0] 
    local_shap = explanation.values[0, :, class_index]
    expected_value = explanation.base_values[0, class_index]
    
    fig_waterfall = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(input_features) + ["total"],
        y=input_features + ["Prediction"],
        x=list(local_shap) + [expected_value + sum(local_shap)],
        text=[f"{val:+.2f}" for val in local_shap] + [f"{expected_value + sum(local_shap):.2f}"],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ff6b6b"}},
        increasing={"marker": {"color": "#4ecdc4"}},
        totals={"marker": {"color": "#feca57"}}
    ))
    
    fig_waterfall.update_layout(
        title=f"Feature Impact on {top_3_crops[0].capitalize()}",
        template="plotly_dark",
        margin=dict(l=100, t=40, b=40),
        waterfallgap=0.3
    )
    
    # Return cards, SHAP figure, and pass the top crop back to the Dropdown UI
    return card1, card2, card3, fig_waterfall, top_3_crops[0]


# -----------------------------------------------------------------------------
# Callback 2B: Real-Time Radar Chart (Triggered by Dropdown OR Sliders)
# -----------------------------------------------------------------------------
@app.callback(
    Output('radar-chart', 'figure'),
    [Input('radar-crop-dropdown', 'value'),
     Input('input-n', 'value'),
     Input('input-p', 'value'),
     Input('input-k', 'value'),
     Input('input-temp', 'value'),
     Input('input-hum', 'value'),
     Input('input-ph', 'value'),
     Input('input-rain', 'value')]
)
def update_radar(selected_crop, n, p, k, temp, hum, ph, rain):
    input_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_values = [n, p, k, temp, hum, ph, rain]
    
    ideal_conditions = crop_means_df[crop_means_df['label'] == selected_crop][input_features].iloc[0].values
    
    fig_radar = go.Figure()
    
    # Add User Input Trace
    fig_radar.add_trace(go.Scatterpolar(
        r=input_values, theta=input_features, fill='toself', name='Your Soil',
        line_color='#feca57'
    ))
    
    # Add Ideal Crop Trace
    fig_radar.add_trace(go.Scatterpolar(
        r=ideal_conditions, theta=input_features, fill='toself', name=f'Ideal {selected_crop.capitalize()}',
        line_color='#4ecdc4', opacity=0.7
    ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, color='#6c757d')),
        showlegend=True,
        template="plotly_dark",
        
        # --- FIX: INCREASED MARGINS drastically to pull labels in ---
        margin=dict(l=70, r=70, t=50, b=50), 
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        
        # Add a title/subtitle to the chart paper for professionalism
        annotations=[
            dict(
                text="Compare Soil Conditions",
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                showarrow=False,
                font=dict(size=14, color="#f8f9fa")
            )
        ]
    )
    
    return fig_radar

# -----------------------------------------------------------------------------
# Callback 3: Sync Sliders to Native Display Boxes
# -----------------------------------------------------------------------------
@app.callback(
    [Output('input-n-readout', 'value'),
     Output('input-p-readout', 'value'),
     Output('input-k-readout', 'value'),
     Output('input-temp-readout', 'value'),
     Output('input-hum-readout', 'value'),
     Output('input-ph-readout', 'value'),
     Output('input-rain-readout', 'value')],
    [Input('input-n', 'value'),
     Input('input-p', 'value'),
     Input('input-k', 'value'),
     Input('input-temp', 'value'),
     Input('input-hum', 'value'),
     Input('input-ph', 'value'),
     Input('input-rain', 'value')]
)
def update_slider_readouts(n, p, k, temp, hum, ph, rain):
    # This natively passes the current slider values directly to the text boxes
    return n, p, k, temp, hum, ph, rain


# -----------------------------------------------------------------------------
# Callbacks for Tab 3 Model Diagnostics
# -----------------------------------------------------------------------------
@app.callback(
    [Output('metric-accuracy', 'children'),
     Output('metric-precision', 'children'),
     Output('metric-recall', 'children'),
     Output('metric-f1', 'children'),
     Output('confusion-matrix-plot', 'figure'),
     Output('stability-report-container', 'children'),
     Output('global-importance-plot', 'figure')],
    [Input('tabs', 'active_tab')]
)
def update_diagnostics_on_load(active_tab):
    # Only execute if the user is looking at Tab 3
    if active_tab != 'tab-3':
        return [dash.no_update] * 7

    # 1. Calculate Global Metrics
    acc = accuracy_score(test_targets, test_preds)
    p, r, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='macro')
    
    def create_metric_card(title, value):
        return [
            html.H6(title, className="text-secondary mb-1"),
            html.H2(f"{value*100:.1f}%", className="text-info fw-bold mb-0")
        ]

    acc_card = create_metric_card("Overall Accuracy", acc)
    p_card = create_metric_card("Precision (Macro Avg)", p)
    r_card = create_metric_card("Recall (Macro Avg)", r)
    f1_card = create_metric_card("F1-Score (Macro Avg)", f1)

    # 2. Confusion Matrix Heatmap (using the ordered capitalized names)
    fig_cm = px.imshow(
        confusion_matrix_data,
        x=ordered_crop_names_capitalized,
        y=ordered_crop_names_capitalized,
        color_continuous_scale="Mint",
        title="Normalized Test Confusion Matrix",
        labels=dict(x="Predicted Crop", y="True Crop"),
        template="plotly_dark",
        aspect="equal"
    )
    
    # Update the text inside the boxes to be smaller
    fig_cm.update_traces(
        text=np.around(confusion_matrix_data, 2), 
        texttemplate="%{text}",
        # We remove the hardcoded small font size here so it scales better
    )

    fig_cm.update_layout(
        coloraxis_showscale=False,
        title_x=0.5,
        # Increase margins to prevent labels from being cut off
        margin=dict(l=120, r=20, t=80, b=120), 
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(ordered_crop_names_capitalized))),
            ticktext=ordered_crop_names_capitalized,
            tickangle=45, # Slant the bottom labels to fit more in
            tickfont=dict(size=9) # Shrunken font for axis labels
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(ordered_crop_names_capitalized))),
            ticktext=ordered_crop_names_capitalized,
            tickfont=dict(size=9) # Shrunken font for axis labels
        )
    )

    # 3. Detailed Stability Table Report
    report_dict = classification_report(test_targets, test_preds, target_names=ordered_crop_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.round(2)
    report_df = report_df.rename(columns={'index': 'Crop', 'precision': 'Precision', 'recall': 'Recall', 'f1-score': 'F1'})
    
    crops_only_report = report_df[report_df['Crop'].isin(ordered_crop_names)].copy()
    crops_only_report['Crop'] = crops_only_report['Crop'].str.capitalize()
    crops_only_report = crops_only_report.drop(columns=['support'])

    # Create a simple, styled Dash table
    stability_table = dbc.Table.from_dataframe(
        crops_only_report,
        striped=True, bordered=True, hover=True,
        className="text-light table-dark table-sm",
        style={"maxHeight": "400px", "display": "block", "overflow-y": "auto", "fontSize": "11px"}
    )
    
    table_layout = [
        html.H5("Detailed Crop Stability Report", className="text-light text-center mb-3"),
        stability_table
    ]

    # 4. Global Feature Importance Bar Chart (Pre-computed)
    fig_global = px.bar(
        global_importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title="Global Feature Importance (All Crops)",
        template="plotly_dark",
        color_discrete_sequence=['#a55eea']
    )
    fig_global.update_layout(margin=dict(l=100, t=50, b=50))
    
    return acc_card, p_card, r_card, f1_card, fig_cm, table_layout, fig_global

if __name__ == '__main__':
    app.run(debug=True)