import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
from pygam import LinearGAM, s

import numpy as np
import scipy.sparse

np.int = int  # 👈 PATCH for deprecated np.int

from pygam import LinearGAM, s

# Patch: ensure any csr_matrix returned behaves like a dense array
scipy.sparse.csr.csr_matrix.A = property(lambda self: self.toarray())
scipy.sparse.csc.csc_matrix.A = property(lambda self: self.toarray())

# Simulate data (replace this with your real data)
df = pd.read_csv('dml_df.csv')
# age = X['Age'].values

# # Prepare dataframe
# df = pd.DataFrame({
#     "Age": age,
#     "Sex": sex,
#     "CATE": tau_hat
# })
df["Sex_str"] = df["Sex"].map({0: "Male", 1: "Female"})

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Explore Heterogeneous Treatment Effects"),

    dcc.Dropdown(
        id='sex-select',
        options=[
            {'label': 'Both', 'value': 'both'},
            {'label': 'Male', 'value': 0},
            {'label': 'Female', 'value': 1},
        ],
        value='both',
        clearable=False
    ),

    dcc.Graph(id='cate-vs-age')
])

@app.callback(
    Output('cate-vs-age', 'figure'),
    Input('sex-select', 'value')
)
def update_figure(sex_val):
    if sex_val == 'both':
        df_plot = df.copy()
        fig = px.scatter(df_plot, x="Age", y="CATE", color="Sex_str", opacity=0.4)

        for label, group_df in df_plot.groupby("Sex"):
            age_arr = group_df['Age'].to_numpy().reshape(-1, 1)
            cate_arr = group_df['CATE'].to_numpy()
            gam = LinearGAM(s(0)).fit(age_arr, cate_arr)
            age_grid = np.linspace(df_plot['Age'].min(), df_plot['Age'].max(), 100)
            y_pred = gam.predict(age_grid)
            y_ci = gam.prediction_intervals(age_grid, width=0.95)

            fig.add_trace(go.Scatter(x=age_grid, y=y_pred,
                                     mode='lines', name=f'{"Male" if label == 0 else "Female"} GAM'))
            fig.add_trace(go.Scatter(x=age_grid, y=y_ci[:, 0],
                                     mode='lines', name=f'{"Male" if label == 0 else "Female"} lower',
                                     line=dict(dash='dash'), showlegend=False))
            fig.add_trace(go.Scatter(x=age_grid, y=y_ci[:, 1],
                                     mode='lines', name=f'{"Male" if label == 0 else "Female"} upper',
                                     line=dict(dash='dash'), showlegend=False))
    else:
        df_plot = df[df['Sex'] == sex_val].copy()
        fig = px.scatter(df_plot, x="Age", y="CATE", opacity=0.4)

        age_arr = df_plot['Age'].to_numpy().reshape(-1, 1)
        cate_arr = df_plot['CATE'].to_numpy()
        gam = LinearGAM(s(0)).fit(age_arr, cate_arr)
        age_grid = np.linspace(df_plot['Age'].min(), df_plot['Age'].max(), 100)
        y_pred = gam.predict(age_grid)
        y_ci = gam.prediction_intervals(age_grid, width=0.95)

        fig.add_trace(go.Scatter(x=age_grid, y=y_pred,
                                 mode='lines', name='GAM Fit'))
        fig.add_trace(go.Scatter(x=age_grid, y=y_ci[:, 0],
                                 mode='lines', name='Lower Bound', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=age_grid, y=y_ci[:, 1],
                                 mode='lines', name='Upper Bound', line=dict(dash='dash')))

    fig.update_layout(title="CATE vs Age with GAM Smoother",
                      xaxis_title="Age", yaxis_title="Estimated dY/dD",
                      template="plotly_white")
    return fig

if __name__ == '__main__':
    app.run(debug=True)

