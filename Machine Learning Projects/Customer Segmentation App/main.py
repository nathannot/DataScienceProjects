import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import io
import base64
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.sparse import issparse
import plotly.graph_objs as go

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Customer Segmentation App"),
    html.H2("Find Optimal Groups for your Marketing Needs"),
    html.H3("Might take a while if data set is large - to speed up process take out unneccessary columns before uploading"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV'),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    dcc.Checklist(
        id='column-checklist',
        options=[],
        value=[],
        inline=True
    ),
    dcc.Graph(id='scatter-plot'),
    html.A(
        id='download-link',
        children=html.Button('Download CSV'),
        href='',
        download='final_data.csv'
    )
])

@app.callback(
    [Output('output-data-upload', 'children'),
     Output('scatter-plot', 'figure'),
     Output('download-link', 'href'),
     Output('column-checklist', 'options')],
    [Input('upload-data', 'contents'),
     Input('column-checklist', 'value')],
    [State('upload-data', 'filename')]
)
def update_output(contents, columns_to_remove, filename):
    if contents is None:
        return [html.Div('Upload a CSV file to see results'), {}, '/', []]

    # Decode the uploaded file content
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(decoded.decode('latin1')))

    # Update column checklist options
    columns = df.columns.tolist()
    column_options = [{'label': col, 'value': col} for col in columns if col != 'User ID']
    
    if columns_to_remove:
        df.drop(columns=columns_to_remove, axis=1, inplace=True, errors='ignore')
    
    # Preprocess
    num_features = df.select_dtypes(include=[float, int])
    cat_features = df.select_dtypes(include=object)
    
    num = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    
    cat = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder()
    )
    
    preprocess = ColumnTransformer([
        ('num', num, make_column_selector(dtype_include=[float, int])),
        ('cat', cat, make_column_selector(dtype_include=object))
    ])
    
    X = preprocess.fit_transform(df)
    
    # Check if X is a sparse matrix
    if issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    # Find optimal number of clusters
    sil = []
    for k in range(3, 20):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_dense)
        sil.append(silhouette_score(X_dense, kmeans.labels_))
    
    best_k = 3 + np.argmax(sil)
    
    best_kmeans = KMeans(n_clusters=best_k, n_init='auto')
    clusters = best_kmeans.fit_predict(X_dense)
    
    model = df.copy()
    
    model['Customer Segment'] = clusters
    if not num_features.empty:
        a = model.groupby('Customer Segment')[num_features.columns].mean().round(2)
    else:
        a=pd.DataFrame()
     # Compute mode for categorical features by cluster
    if not cat_features.empty:
        b = model.groupby('Customer Segment')[cat_features.columns].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.Series(dtype='object'))
    else:
        b=pd.DataFrame()
    if not a.empty and not b.empty:
        final = pd.concat([a, b], axis=1).reset_index()
    elif not a.empty:
        final = a.reset_index()
    elif not b.empty:
        final = b.reset_index()
    else:
        final = pd.DataFrame()
    
    pca = PCA(n_components=2, random_state=42)
    Xt = pca.fit_transform(X_dense)
    
    kmeans_pca = KMeans(n_clusters=best_k, n_init='auto')
    kmeans_pca.fit(Xt)
    clusters_pca = kmeans_pca.labels_
    cc = kmeans_pca.cluster_centers_
    
    # Create the plotly scatter plot
    scatter_data = []
    unique_clusters = np.unique(clusters_pca)
    for cluster in unique_clusters:
        scatter_data.append(go.Scatter(
            x=Xt[clusters_pca == cluster, 0],
            y=Xt[clusters_pca == cluster, 1],
            mode='markers',
            marker=dict(size=10),
            name=f'Cluster {cluster}'
        ))
    
    scatter_data.append(go.Scatter(
        x=cc[:, 0],
        y=cc[:, 1],
        mode='markers',
        marker=dict(color='red', size=25, symbol='x'),
        name='Cluster Centres'
    ))
    
    layout = go.Layout(
        xaxis=dict(title='Principal Component 1'),
        yaxis=dict(title='Principal Component 2'),
        title='Customer Segments Visualized',
        showlegend=True
    )
    
    figure = {'data': scatter_data, 'layout': layout}
    
    
    # Save the final DataFrame as a CSV file
    csv_string = final.to_csv(index=False, encoding='utf-8')
    csv_bytes = csv_string.encode()
    b64_csv = base64.b64encode(csv_bytes).decode()
    href = f"data:text/csv;base64,{b64_csv}"
    
    return dash_table.DataTable(
        data=final.reset_index().to_dict('records'),
        columns=[{'name': c, 'id': c} for c in final.reset_index().columns],
        style_table={'overflowX': 'auto'}
    ), figure, href, column_options

if __name__ == '__main__':
    app.run_server(debug=True)
