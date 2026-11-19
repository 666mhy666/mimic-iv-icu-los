"""Archived coursework Dash interface for locally stored, trusted model files."""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import base64
import io
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report

import dash
from dash import dcc, html, Input, Output, State, dash_table


MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILES = {
    "Logistic Regression": "logit_model.pkl",
    "Random Forest": "rf_model.pkl",
    "Boosting": "xgb_model.pkl",
}


def load_course_models():
    models = {}
    for label, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Original model missing: {path}. See app/README.md for local setup."
            )
        # Only load artifacts produced locally by the coursework notebook.
        with path.open("rb") as stream:
            models[label] = pickle.load(stream)
    return models


models = load_course_models()

numeric_features = [
    'anchor_age', 'wbc', 'potassium', 'sodium', 'creatinine', 'bicarbonate', 'hematocrit',
    'chloride', 'glucose', 'heart_rate', 'temperature_fahrenheit',
    'respiratory_rate', 'non_invasive_blood_pressure_systolic',
    'non_invasive_blood_pressure_diastolic'
]
categorical_features = ['gender', 'marital_status', 'race']
input_features = numeric_features + categorical_features

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "ICU Long Stay Prediction App"

app.layout = html.Div([
    html.H2("ICU Long Stay Prediction App"),

    dcc.RadioItems(
        id='mode',
        options=[
            {'label': 'Prediction Mode', 'value': 'predict'},
            {'label': 'Evaluation Mode', 'value': 'evaluate'}
        ],
        value='predict',
        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
    ),

    html.Label("Select Model(s):"),
    dcc.Checklist(
        id='model-select',
        options=[{'label': k, 'value': k} for k in models],
        value=["Logistic Regression"],
        labelStyle={'display': 'inline-block', 'margin-right': '20px'}
    ),

    # both controls are here, but we’ll hide one or the other
    html.Div(id='predict-controls', children=[
        html.Br(),
        html.Label("Upload a CSV file (optional):"),
        dcc.Upload(
            id='upload-predict',
            children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
            multiple=False
        ),
        html.Br(),
        html.Label("Or manually input values:"),
        html.Div([ html.Div([html.Label(col), dcc.Input(id=col, type='text')])
                  for col in input_features ]),
    ]),

    html.Div(id='eval-controls', children=[
        html.Br(),
        html.Label("Upload .parquet file for evaluation:"),
        dcc.Upload(
            id='upload-eval',
            children=html.Div(['Drag and Drop or ', html.A('Select Parquet File')]),
            multiple=False
        ),
    ], style={'display': 'none'}),    # start hidden

    html.Br(),
    html.Button("Go", id="action-button", n_clicks=0),

    html.Div(id='output-area')
])


@app.callback(
    Output('predict-controls', 'style'),
    Output('eval-controls',   'style'),
    Input('mode', 'value'),
)
def swap_mode_ui(mode):
    if mode == 'predict':
        return {'display':'block'}, {'display':'none'}
    else:
        return {'display':'none'},  {'display':'block'}


@app.callback(
    Output('output-area', 'children'),
    Input('action-button', 'n_clicks'),
    State('mode',         'value'),
    State('upload-predict','contents'),
    State('upload-eval',   'contents'),
    State('model-select',  'value'),
    [State(col, 'value') for col in input_features],
    prevent_initial_call=True
)
def handle_action(n, mode, csv_contents, pq_contents, selected_models, *manual_inputs):
    from dash import html

    if not selected_models:
        return html.Div("Please select at least one model.")

    # -------------------- Prediction Mode --------------------
    if mode == 'predict':
        # load df from CSV or manual inputs...
        if csv_contents:
            content_type, content_string = csv_contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            except Exception as e:
                return html.Div(f"Failed to read uploaded CSV: {e}")
        else:
            # build df from manual_inputs (same as before)...
            data = dict(zip(input_features, manual_inputs))
            for col in categorical_features:
                if not data[col]:
                    data[col] = 'Unknown'
            for col in numeric_features:
                try:
                    data[col] = float(data[col])
                except:
                    data[col] = np.nan
            df = pd.DataFrame([data])

        missing = sorted(set(input_features) - set(df.columns))
        if missing:
            return html.Div("Missing predictors: " + ", ".join(missing))
        df = df[input_features]
        outputs = []
        for model_name in selected_models:
            model = models[model_name]
            try:
                preds = model.predict(df)
                probs = model.predict_proba(df)[:,1]
            except Exception as e:
                return html.Div(f"Model prediction failed: {str(e)}")

            result_df = df[input_features].copy()
            result_df['Predicted Label'] = preds
            result_df['Probability'] = probs

            outputs.append(html.H4(f"{model_name} Predictions"))
            outputs.append(
                dash_table.DataTable(
                    data=result_df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in result_df.columns],
                    style_table={'overflowX': 'auto'}
                )
            )
        return outputs

    # -------------------- Evaluation Mode --------------------
    else:
        if not pq_contents:
            return html.Div("Please upload a Parquet file for evaluation.")
        content_type, content_string = pq_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_parquet(io.BytesIO(decoded))
        except Exception as e:
            return html.Div([
                html.P("Failed to read .parquet file."),
                html.Pre(str(e))
            ])

        try:
            X = df[input_features].copy()
            y = df['los_long']
        except KeyError as e:
            return html.Div(f"Missing required columns: {e}")

        outputs = []
        for model_name in selected_models:
            model = models[model_name]
            try:
                y_pred = model.predict(X)
                y_prob = model.predict_proba(X)[:, 1]
                acc = accuracy_score(y, y_pred)
                auc = roc_auc_score(y, y_prob)
                fpr, tpr, _ = roc_curve(y, y_prob)
                cm = confusion_matrix(y, y_pred)

                outputs.append(html.H4(f"{model_name} Evaluation"))
                outputs.append(html.P(f"Accuracy: {acc:.3f}, AUC: {auc:.3f}"))

                # ROC Curve
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                            name='Random', line=dict(dash='dash')))
                roc_fig.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
                outputs.append(dcc.Graph(figure=roc_fig))

                # Confusion Matrix
                # 1) show a text classification report
                target_names = ['No Long Stay','Long Stay']
                report = classification_report(y, y_pred, target_names=target_names)
                outputs.append(html.Pre(report, style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'}))

                # 2) Build a labelled DataFrame for the confusion matrix
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"True: {n}" for n in target_names],
                    columns=[f"Pred: {n}" for n in target_names]
                ).reset_index().rename(columns={'index': ''})

                # 3) Render it as a Dash DataTable
                outputs.append(html.H5("Confusion Matrix"))
                outputs.append(
                    dash_table.DataTable(
                        data=cm_df.to_dict('records'),
                        columns=[{'name': c, 'id': c} for c in cm_df.columns],
                        style_header={'fontWeight': 'bold'},
                        style_cell={'textAlign': 'center'},
                        style_data_conditional=[
                            {'if': {'row_index': 'even'}, 'backgroundColor': '#f9f9f9'}
                        ],
                        style_table={'overflowX': 'auto'}
                    )
                )


            except Exception as e:
                outputs.append(html.Div(f"Evaluation failed for {model_name}: {e}"))

        return outputs


if __name__ == '__main__':
    app.run(host="127.0.0.1", debug=False)
