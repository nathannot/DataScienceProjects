import pandas as pd
import numpy as np

data = pd.read_csv('housing.csv')
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

#create a function to create ratios and age_cat and drop irrelevant columns
def ratio(X):
    X['rooms_per_house']=X['total_rooms']/X['households']
    X['bedrooms_ratio']=X['total_bedrooms']/X['total_rooms']
    X['age_cat']=pd.cut(X['housing_median_age'], bins=[0,18,29,37,50,np.inf],
                          labels = [1,2,3,4,5])
    return X.drop(['total_rooms','housing_median_age'],axis=1)


#create log pipeline
log_pipeline = make_pipeline(
    FunctionTransformer(np.log,inverse_func=np.exp, feature_names_out="one-to-one"),
    SimpleImputer(strategy='median'),
    StandardScaler()
)

#create cat pipeline
cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown= 'ignore')
)

#create default pipeline
def_pipeline = make_pipeline(
    StandardScaler()
)

#combine all into one
feat_eng = ColumnTransformer([
    ("log", log_pipeline, ["total_bedrooms","bedrooms_ratio", "rooms_per_house", "population",
                               "households", "median_income"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
], remainder = def_pipeline)

#first define feature and target sets
y = data['median_house_value']
X = ratio(data).drop('median_house_value',axis=1)

#Create final train and test sets
#apply feat eng pipeline on X
from sklearn.model_selection import train_test_split
X_feat = feat_eng.fit_transform(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(X_feat,y,test_size=0.2, random_state=42,
                                                stratify = X.age_cat)

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

bgr = BaggingRegressor(estimator=HistGradientBoostingRegressor(),
                      n_estimators=40,
                      max_features=0.7,
                      random_state=0)


gbr = HistGradientBoostingRegressor(max_iter=200,
                                    quantile = 0.1,
                                   loss='absolute_error',
                                   learning_rate=0.1,
                                   l2_regularization=0)

#fit best model to see performance

rfr = RandomForestRegressor(n_estimators=100,
                           min_samples_split=2,
                           max_features=None,
                           criterion='poisson',
                           random_state=0)

#build voting regressor for 3 optimal models
from sklearn.ensemble import VotingRegressor

#initial model
estimators = [
    ('rf',rfr),
    ('bg',bgr),
    ('hgb',gbr)
]
vote = VotingRegressor(estimators=estimators)
                          
vote_model = vote.fit(Xtrain, ytrain)

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc

# Initialize the Dash app
app = dash.Dash(__name__)

# Define feature columns
feature_columns = [
    'log__total_bedrooms', 'log__bedrooms_ratio', 'log__rooms_per_house',
    'log__population', 'log__households', 'log__median_income',
    'cat__ocean_proximity_<1H OCEAN', 'cat__ocean_proximity_INLAND',
    'cat__ocean_proximity_ISLAND', 'cat__ocean_proximity_NEAR BAY',
    'cat__ocean_proximity_NEAR OCEAN', 'remainder__longitude',
    'remainder__latitude', 'remainder__housing_median_age_cat'
]


# Create the app layout
app.layout = html.Div([
    html.Div([
        html.H1("California Real Estate Price Prediction", style={'text-align': 'center'}),
        
        html.Div([
            dcc.Input(id='total_bedrooms', type='number', placeholder='Total Bedrooms',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='total_rooms', type='number', placeholder='Total Rooms',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='population', type='number', placeholder='Population',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='households', type='number', placeholder='Households',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='median_income', type='number', placeholder='Median Income',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='longitude', type='number', placeholder='Longitude',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='latitude', type='number', placeholder='Latitude',
                      style={'margin': '10px', 'padding': '10px'}),
            
            dcc.Dropdown(
                id='ocean_proximity',
                options=[
                    {'label': 'Less than 1H Ocean', 'value': 'cat__ocean_proximity_<1H OCEAN'},
                    {'label': 'Inland', 'value': 'cat__ocean_proximity_INLAND'},
                    {'label': 'Island', 'value': 'cat__ocean_proximity_ISLAND'},
                    {'label': 'Near Bay', 'value': 'cat__ocean_proximity_NEAR BAY'},
                    {'label': 'Near Ocean', 'value': 'cat__ocean_proximity_NEAR OCEAN'}
                ],
                placeholder='Select Ocean Proximity',
                style={'margin': '10px', 'padding': '10px'}
            ),
            
            dcc.Dropdown(
                id='age_cat',
                options=[
                    {'label': '0-18', 'value': 1},
                    {'label': '19-28', 'value': 2},
                    {'label': '29-38', 'value': 3},
                    {'label': '39-48', 'value': 4},
                    {'label': '49+', 'value': 5}
                ],
                placeholder='Select Age Category',
                style={'margin': '10px', 'padding': '10px'}
            ),
            
            html.Button('Predict Price', id='predict_button', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px', 'background-color': '#007BFF', 'color': 'white'}),
            html.Button('Load Example Data', id='example_button', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px', 'background-color': '#28a745', 'color': 'white'}),
        ], style={'text-align': 'center'}),
        
        html.Div(id='prediction_output', style={'text-align': 'center', 'font-size': '20px', 'margin-top': '20px'})
    ], style={'width': '50%', 'margin': '0 auto', 'border': '2px solid #007BFF', 'padding': '20px', 'border-radius': '10px'})
])

# Define callback to update output
@app.callback(
    [Output('total_bedrooms', 'value'),
     Output('total_rooms', 'value'),
     Output('population', 'value'),
     Output('households', 'value'),
     Output('median_income', 'value'),
     Output('longitude', 'value'),
     Output('latitude', 'value'),
     Output('ocean_proximity', 'value'),
     Output('age_cat', 'value'),
     Output('prediction_output', 'children')],
    [Input('predict_button', 'n_clicks'), Input('example_button', 'n_clicks')],
    [State('total_bedrooms', 'value'),
     State('total_rooms', 'value'),
     State('population', 'value'),
     State('households', 'value'),
     State('median_income', 'value'),
     State('longitude', 'value'),
     State('latitude', 'value'),
     State('ocean_proximity', 'value'),
     State('age_cat', 'value')]
)
def update_output(predict_clicks, example_clicks, total_bedrooms, total_rooms, population, households, median_income, longitude, latitude, ocean_proximity, age_cat):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'example_button':
        return 129, 880, 322, 126, 8.3252, -122.23, 37.88, 'cat__ocean_proximity_NEAR BAY', 4, ''
    elif button_id == 'predict_button' and all(v is not None for v in [total_bedrooms, total_rooms, population, households, median_income, longitude, latitude, ocean_proximity, age_cat]):
        # Calculate log-transformed features
        log_total_bedrooms = np.log(total_bedrooms)
        log_bedrooms_ratio = np.log(total_bedrooms / (total_rooms))
        log_rooms_per_house = np.log(total_rooms / (households))
        log_population = np.log(population)
        log_households = np.log(households)
        log_median_income = np.log(median_income)
        
        # Binning for age_cat
        age_bins = [0, 18, 28, 38, 48, np.inf]
        age_labels = [1, 2, 3, 4, 5]
        age_cat_value = pd.cut([age_cat], bins=age_bins, labels=age_labels)[0]
        
        # Create the dictionary to represent ocean_proximity as one-hot encoded values
        ocean_proximity_dict = {
            'cat__ocean_proximity_<1H OCEAN': 0,
            'cat__ocean_proximity_INLAND': 0,
            'cat__ocean_proximity_ISLAND': 0,
            'cat__ocean_proximity_NEAR BAY': 0,
            'cat__ocean_proximity_NEAR OCEAN': 0
        }
        ocean_proximity_dict[ocean_proximity] = 1

        # Create the features DataFrame with all columns
        features = pd.DataFrame([[log_total_bedrooms, log_bedrooms_ratio, log_rooms_per_house, log_population, 
                                  log_households, log_median_income,
                                  ocean_proximity_dict['cat__ocean_proximity_<1H OCEAN'],
                                  ocean_proximity_dict['cat__ocean_proximity_INLAND'],
                                  ocean_proximity_dict['cat__ocean_proximity_ISLAND'],
                                  ocean_proximity_dict['cat__ocean_proximity_NEAR BAY'],
                                  ocean_proximity_dict['cat__ocean_proximity_NEAR OCEAN'],
                                  longitude, latitude, age_cat_value]], 
                                columns=feature_columns)
        
        # Predict
        prediction = vote_model.predict(features)[0]
        return total_bedrooms, total_rooms, population, households, median_income, longitude, latitude, ocean_proximity, age_cat, f'Predicted House Price: ${prediction:.2f}'
    elif predict_clicks > 0:
        return total_bedrooms, total_rooms, population, households, median_income, longitude, latitude, ocean_proximity, age_cat, 'Please enter all values to get a prediction'
    return total_bedrooms, total_rooms, population, households, median_income, longitude, latitude, ocean_proximity, age_cat, ''

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

