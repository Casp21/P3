import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import keras

# Cargar los datos y modelos
data = pd.read_csv('C:/Users/camil/Downloads/Analitica computacional/Proyecto_3/data_dash_2.csv')
data_num = pd.read_csv('C:/Users/camil/Downloads/Analitica computacional/Proyecto_3/data_num.csv')
model = keras.models.load_model('C:/Users/camil/Downloads/Analitica computacional/Proyecto_3/modelo_p3.keras')

# Diccionarios y listas útiles
friendly_names = {
    'punt_global': 'Puntaje Global',
    'desemp_ingles': 'Desempeño en Inglés',
    'fami_educacionmadre': 'Educación de la Madre',
    'fami_educacionpadre': 'Educación del Padre',
    'fami_estratovivienda': 'Estrato de la Vivienda',
    'fami_tienecomputador': 'Tiene Computadora',
    'fami_tieneinternet': 'Tiene Internet',
    'fami_tieneautomovil': 'Tiene Automóvil'
}
columns = list(friendly_names.keys())

unique_values = {
    'cole_naturaleza': data['cole_naturaleza'].unique().tolist(),
    'fami_educacionmadre': data['fami_educacionmadre'].unique().tolist(),
    'fami_educacionpadre': data['fami_educacionpadre'].unique().tolist(),
    'fami_estratovivienda': data['fami_estratovivienda'].unique().tolist(),
    'fami_tienecomputador': data['fami_tienecomputador'].unique().tolist(),
    'fami_tieneinternet': data['fami_tieneinternet'].unique().tolist(),
    'fami_tieneautomovil': data['fami_tieneautomovil'].unique().tolist(),
    'desemp_ingles': data['desemp_ingles'].unique().tolist()
}

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout de la aplicación
app.layout = dbc.Container([
    html.H1("Dashboard Integrado para Análisis y Predicción de Pruebas Saber 11"),
    
    # Sección de análisis de datos
    dbc.Row([
        dbc.Col(html.Div([
            html.H2("Análisis de Pruebas Saber 11"),
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Selecciona la característica para el eje X:"),
                            dcc.Dropdown(
                                id='x-axis-feature',
                                options=[
                                    {'label': 'Estrato Socioeconómico', 'value': 'fami_estratovivienda'},
                                    {'label': 'Género', 'value': 'estu_genero'},
                                    {'label': 'Colegio Bilingüe', 'value': 'cole_bilingue'},
                                    {'label': 'Naturaleza del Colegio', 'value': 'cole_naturaleza'},
                                    {'label': 'Calendario del Colegio', 'value': 'cole_calendario'}
                                ],
                                value='fami_estratovivienda'
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Selecciona el puntaje para el eje Y:"),
                            dcc.Dropdown(
                                id='y-axis-feature',
                                options=[
                                    {'label': 'Puntaje de Inglés', 'value': 'punt_ingles'},
                                    {'label': 'Puntaje Global', 'value': 'punt_global'}
                                ],
                                value='punt_global'
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Selecciona el rango de puntaje:"),
                            dcc.RangeSlider(
                                id='score-range',
                                min=0,
                                max=500,
                                step=1,
                                value=[0, 500],
                                marks={i: str(i) for i in range(0, 501, 50)}
                            )
                        ], width=4)
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='bar-chart'), width=6),
                        dbc.Col(dcc.Graph(id='pie-chart'), width=6)
                    ]),
                    html.Hr(),
                    html.Label("Selecciona las variables para la matriz de correlación:"),
                    dbc.Card(
                        dbc.CardBody(
                            dbc.Checklist(
                                id='column-checklist',
                                options=[{'label': friendly_names[col], 'value': col} for col in columns],
                                value=columns,  # Preseleccionar todas las columnas por defecto
                                inline=True,
                                switch=True,
                            )
                        )
                    ),
                    dcc.Graph(id='heatmap')
                ])
            )
        ]), width=12)
    ]),
    
    html.Hr(),
    
    # Sección de predicción
    dbc.Row([
        dbc.Col(html.Div([
            html.H2("Predicción del Puntaje Global de la Prueba Saber 11"),
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Naturaleza del Colegio:"),
                            dcc.Dropdown(id='cole_naturaleza', options=[{'label': k, 'value': k} for k in unique_values['cole_naturaleza']], value=unique_values['cole_naturaleza'][0])
                        ], width=4),
                        dbc.Col([
                            html.Label("Educación de la Madre:"),
                            dcc.Dropdown(id='fami_educacionmadre', options=[{'label': k, 'value': k} for k in unique_values['fami_educacionmadre']], value=unique_values['fami_educacionmadre'][0])
                        ], width=4),
                        dbc.Col([
                            html.Label("Educación del Padre:"),
                            dcc.Dropdown(id='fami_educacionpadre', options=[{'label': k, 'value': k} for k in unique_values['fami_educacionpadre']], value=unique_values['fami_educacionpadre'][0])
                        ], width=4),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Estrato de la Vivienda:"),
                            dcc.Dropdown(id='fami_estratovivienda', options=[{'label': k, 'value': k} for k in unique_values['fami_estratovivienda']], value=unique_values['fami_estratovivienda'][0])
                        ], width=4),
                        dbc.Col([
                            html.Label("¿Tiene Computadora?:"),
                            dcc.Dropdown(id='fami_tienecomputador', options=[{'label': k, 'value': k} for k in unique_values['fami_tienecomputador']], value=unique_values['fami_tienecomputador'][0])
                        ], width=4),
                        dbc.Col([
                            html.Label("¿Tiene Internet?:"),
                            dcc.Dropdown(id='fami_tieneinternet', options=[{'label': k, 'value': k} for k in unique_values['fami_tieneinternet']], value=unique_values['fami_tieneinternet'][0])
                        ], width=4),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("¿Tiene Automóvil?:"),
                            dcc.Dropdown(id='fami_tieneautomovil', options=[{'label': k, 'value': k} for k in unique_values['fami_tieneautomovil']], value=unique_values['fami_tieneautomovil'][0])
                        ], width=4),
                        dbc.Col([
                            html.Label("Desempeño en Inglés:"),
                            dcc.Dropdown(id='desemp_ingles', options=[{'label': k, 'value': k} for k in unique_values['desemp_ingles']], value=unique_values['desemp_ingles'][0])
                        ], width=4),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Predecir Puntaje", id='predict-button', color='primary')
                        ], width=4),
                    ]),
                ])
            ),
            dbc.Row([
                dbc.Col(html.Div(id='prediction-output', style={'margin-top': '20px'}), width=12)
            ])
        ]), width=12)
    ])
], fluid=True)

# Callback para actualizar la gráfica de barras y la gráfica de tortas
@app.callback(
    [Output('bar-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('score-range', 'max'),
     Output('score-range', 'marks')],
    [Input('x-axis-feature', 'value'),
     Input('y-axis-feature', 'value'),
     Input('score-range', 'value')]
)
def update_graphs(x_feature, y_feature, score_range):
    # Ajustar el valor máximo permitido y las marcas en el control deslizante
    if y_feature == 'punt_global':
        max_score = 500
        marks = {i: str(i) for i in range(0, max_score + 1, 50)}
    else:
        max_score = 100
        marks = {i: str(i) for i in range(0, max_score + 1, 10)}

    # Agrupar y calcular el promedio de los puntajes
    data_grouped = data.groupby(x_feature)[y_feature].mean().reset_index()

    bar_fig = px.bar(
        data_grouped,
        x=x_feature,
        y=y_feature,
        color=x_feature,
        barmode='group',
        height=500,
        title=f'Promedio de {y_feature.replace("_", " ").capitalize()} por {x_feature.replace("_", " ").capitalize()}'
    )

    # Calcular el porcentaje de puntajes dentro del rango ingresado
    filtered_data = data[(data[y_feature] >= score_range[0]) & (data[y_feature] <= score_range[1])]
    pie_data = filtered_data[x_feature].value_counts(normalize=True).reset_index()
    pie_data.columns = [x_feature, 'percentage']

    pie_fig = go.Figure(data=[go.Pie(
        labels=pie_data[x_feature],
        values=pie_data['percentage'],
        hole=.3
    )])
    pie_fig.update_layout(
        title=f'Porcentaje de {y_feature.replace("_", " ").capitalize()} entre {score_range[0]} y {score_range[1]}'
    )

    return bar_fig, pie_fig, max_score, marks

# Callback para actualizar el heatmap
@app.callback(
    Output('heatmap', 'figure'),
    [Input('column-checklist', 'value')]
)
def update_heatmap(selected_columns):
    if len(selected_columns) < 2:
        return go.Figure()  # Devuelve una figura vacía si no hay suficientes columnas seleccionadas

    # Calcular la matriz de correlación para las columnas seleccionadas
    correlation_matrix = data_num[selected_columns].corr()

    # Crear el heatmap
    heatmap = go.Heatmap(
        z=correlation_matrix.values,
        x=[friendly_names[col] for col in correlation_matrix.columns],
        y=[friendly_names[col] for col in correlation_matrix.index],
        colorscale='Viridis',
        zmin=-1, zmax=1,
        hoverongaps=False
    )

    # Añadir anotaciones de texto
    annotations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.index)):
            annotations.append(
                go.layout.Annotation(
                    text=str(round(correlation_matrix.values[i][j], 2)),
                    x=friendly_names[correlation_matrix.columns[i]],
                    y=friendly_names[correlation_matrix.index[j]],
                    xref='x1', yref='y1',
                    showarrow=False,
                    font=dict(color='black')
                )
            )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title='Matriz de Correlación entre Variables Seleccionadas',
        annotations=annotations,
        xaxis_nticks=36
    )

    return fig

# Callback para predecir el puntaje global
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('cole_naturaleza', 'value'),
     State('fami_educacionmadre', 'value'),
     State('fami_educacionpadre', 'value'),
     State('fami_estratovivienda', 'value'),
     State('fami_tienecomputador', 'value'),
     State('fami_tieneinternet', 'value'),
     State('fami_tieneautomovil', 'value'),
     State('desemp_ingles', 'value')]
)
def predict_score(n_clicks, *inputs):
    if n_clicks is not None:
        # Procesar los inputs para el modelo
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), list(unique_values.keys()))  # Asumiendo que todos los inputs son categóricos
            ]
        )
        processed_inputs = preprocessor.fit_transform(np.array(inputs).reshape(1, -1))

        prediction = model.predict(processed_inputs)[0][0]
        return dbc.Alert(f'El puntaje global de la prueba predicho es: {prediction:.2f}', color='success')
    return ""

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
