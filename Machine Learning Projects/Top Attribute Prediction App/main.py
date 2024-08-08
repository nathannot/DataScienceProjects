import dash
from dash import dcc, html, Input, Output, callback, dash_table
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import io
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Initialize the Dash app
app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.H1("Identify Key Attributes in Your Data Using Machine Learning",
            style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#333', 'fontFamily': 'Arial'}),
    
    # Description and instructions
    html.Div([
        html.P("Upload your CSV data and specify the ID column (if there is one) and class column name (column you want to find top attributes for). Then press process data and the app will then identify and display the top attributes based on their importance.",
               style={'textAlign': 'center', 'marginBottom': '20px', 'fontSize': '18px', 'color': '#666', 'fontFamily': 'Arial'})
    ], style={'backgroundColor': '#e8f4f8', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '30px'}),

    # Container for input fields and buttons
    html.Div([
        # Input for ID column
        html.Div([
            html.Label("ID Column:", style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dcc.Input(id='id-column', type='text', placeholder='Enter ID column name (optional)', style={'padding': '10px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc', 'width': '80%'})
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Input for class column
        html.Div([
            html.Label("Class Column:", style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dcc.Input(id='class-column', type='text', placeholder='Enter class column name', style={'padding': '10px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ccc', 'width': '80%'})
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),

        # File upload component
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload CSV', style={
                    'padding': '12px 24px',
                    'fontSize': '16px',
                    'backgroundColor': '#007bff',
                    'color': '#fff',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'transition': 'background-color 0.3s',
                    'marginBottom': '20px'
                }),
                multiple=False
            )
        ], style={'textAlign': 'center'}),

        # Button to trigger processing
        html.Div([
            html.Button('Process Data', id='process-button', n_clicks=0, style={
                'padding': '12px 24px',
                'fontSize': '16px',
                'backgroundColor': '#28a745',
                'color': '#fff',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'transition': 'background-color 0.3s',
                'marginBottom': '20px'
            })
        ], style={'textAlign': 'center'}),

        # Loading spinner
        dcc.Loading(
            id="loading",
            type="circle",
            children=[html.Div(id='loading-output')],
            style={'textAlign': 'center'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),

    # Display model accuracy
    html.Div(id='accuracy-container', style={'textAlign': 'center', 'marginBottom': '30px', 'fontSize': '18px', 'color': '#333', 'fontFamily': 'Arial'}),

    # Display table of feature importances
    html.Div(id='table-container', style={'marginBottom': '30px', 'display': 'flex', 'justifyContent': 'center'}),

    # Download button
    html.Div([
        html.Button('Download CSV', id='download-button', n_clicks=0, style={
            'padding': '12px 24px',
            'fontSize': '16px',
            'backgroundColor': '#007bff',
            'color': '#fff',
            'border': 'none',
            'borderRadius': '5px',
            'cursor': 'pointer',
            'transition': 'background-color 0.3s',
            'marginBottom': '20px'
        }),
        dcc.Download(id='download-data')
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),

    # Display plot of feature importances
    dcc.Graph(id='feature-importance-plot', style={'height': '500px', 'width': '80%', 'margin': '0 auto'})
], style={'backgroundColor': '#f0f8ff', 'padding': '30px', 'borderRadius': '15px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'textAlign': 'center'})

# Callback for uploading and processing CSV file
@callback(
    Output('accuracy-container', 'children'),
    Output('table-container', 'children'),
    Output('feature-importance-plot', 'figure'),
    Output('download-data', 'data'),
    Output('loading-output', 'children'),
    Input('upload-data', 'contents'),
    Input('class-column', 'value'),
    Input('id-column', 'value'),
    Input('process-button', 'n_clicks'),
    Input('download-button', 'n_clicks')
)
def update_output(uploaded_file, class_column, id_column, process_n_clicks, download_n_clicks):
    if uploaded_file is None or not class_column or process_n_clicks == 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Decode the uploaded CSV file
    content_type, content_string = uploaded_file.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    if class_column not in df.columns:
        return "Class column not found", dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Remove ID column if specified
    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])

    def neg_shift(x):
        # Create a copy of x to avoid modifying the original data
        x = x.copy()
    
        # Iterate over each column
        for i in range(x.shape[1]):
            # Check if the minimum value of the column is negative
            if np.min(x[:, i]) < 0:
                # Shift the column to make all values non-negative
                x[:, i] = x[:, i] - np.min(x[:, i])
        
        return x

    num_pipeline = make_pipeline(
        SimpleImputer(strategy = 'median'),
        FunctionTransformer(neg_shift, feature_names_out='one-to-one'),
        FunctionTransformer(np.log1p, feature_names_out='one-to-one'),
        StandardScaler()
    )

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder()
    )

    preprocess = ColumnTransformer([
        ('num', num_pipeline, make_column_selector(dtype_include=[int, float])),
        ('cat', cat_pipeline, make_column_selector(dtype_include=object))
    ])

   

    
    X = preprocess.fit_transform(df.drop([class_column],axis=1))
    y = df[class_column]
    
    # Handle missing values in the target variable if needed
    if y.isnull().any():
        si = SimpleImputer(strategy='most_frequent')
        y = si.fit_transform(y.values.reshape(-1, 1)).ravel()  # Reshape for imputation and then flatten

    # Encode the target variable if it's categorical
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    accuracy = rf.score(X_test, y_test)
    feat_imp = rf.feature_importances_
    columns = preprocess.get_feature_names_out()
    importance = pd.DataFrame({'Features': columns, 'Importance': feat_imp}).sort_values('Importance', ascending=False).reset_index().drop('index', axis=1).round(3).head(10)
    importance['Features'] = importance['Features'].str[5:]
    
    # Create Plotly figure
    fig = px.bar(importance, x='Importance', y='Features', title='Top Attributes', labels={'Importance': 'Importance Value', 'Features': 'Attribute'}, orientation='h')
    fig.update_layout(yaxis_title='Attribute', xaxis_title='Importance Value', yaxis=dict(categoryorder='total ascending'))

    # Create DataTable with centered text and equal column widths
    table = dash_table.DataTable(
        importance.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in importance.columns],
        style_table={'overflowX': 'auto'},
        style_header={
            'fontWeight': 'bold',
            'textAlign': 'center',
            'padding': '10px'
        },
        style_cell={
            'textAlign': 'center',
            'padding': '10px'
        }
    )

    accuracy_message = f"Model Accuracy: {accuracy:.2f}"
    
    # Convert DataFrame to CSV for download
    if download_n_clicks > 0:
        csv_data = importance.to_csv(index=False)
        return accuracy_message, table, fig, dict(content=csv_data, filename='results.csv'), dash.no_update

    return accuracy_message, table, fig, dash.no_update, dash.no_update

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

