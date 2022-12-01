import pickle
import sklearn
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
from datetime import date
import dash_mantine_components as dmc
import time
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from functions import apply_model, fraud_classification
from dash.exceptions import PreventUpdate
from dash.dash import no_update

# transformed data
data = pd.read_csv("transformed.csv")
data['job'] = data['job'].str.title()

# Belfius Logo
image_path = 'assets/Logo.jpg'

# load the model created in databricks
rf_model = pickle.load(open('randomforest.pkl', 'rb'))

# category of the merchant
category_dropdown = dcc.Dropdown(id="category",
                        options=[
                            {"label": 'Entertainment', 'value': 'entertainment'},
                            {"label": 'Food & Dining', 'value': 'food_dining'},
                            {"label": 'Gas transportation', 'value': 'gas_transport'},
                            {"label": 'Grocery stores (Net)', 'value': 'grocery_net'},
                            {"label": 'Grocery stores (POS: point of sale)', 'value': 'grocery_pos'},
                            {"label": 'Health & Fitness', 'value': 'health_fitness'},
                            {"label": 'Home', 'value': 'home'},
                            {"label": 'Pets', 'value': 'kids_pets'},
                            {"label": 'Others (Net)', 'value': 'misc_net'},
                            {"label": 'Others (POS: point of sale)', 'value': 'misc_pos'},
                            {"label": 'Personal Care', 'value': 'personal_care'},
                            {"label": 'Shopping stores (Net)', 'value': 'shopping_net'},
                            {"label": 'Shopping stores (POS: point of sale)', 'value': 'shopping_pos'},
                            {"label": 'Travel', 'value': 'travel'}
                        ],
                            value='entertainment', style={"color": "black", 'text-align':'center'},
                                 placeholder= 'Select the category')

# Job of credit card holder
job_dropdown = dcc.Dropdown(data['job'].unique(),id="job",
                            placeholder='Select the type of job', value= 'Mining Engineer',
                            style={"color": "black", 'text-align':'center'})

# DOB of credit card holder
dob_dropdown = dcc.DatePickerSingle(id = 'dob',
    month_format='MMMM Y',
    placeholder='Select the date',
    display_format= 'YYYY/MM/DD',
    day_size=30,
    style = {'width':'300px','font-size':12, 'color':'#415c6c'}
)
# Time of the transaction
time_dropdown = dmc.TimeInput(
            style={"width": 300}, value='09:32:30',
            withSeconds=True,
        )



app = dash.Dash(__name__, title="Predictive model",external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container(
    [   html.Div(html.Img(src= image_path, style={'height':'10%', 'width':'10%', 'align': 'left'}),
                style={'background-color': '#c30045'}),
        html.H1(children='Fraud Prediction Model',
                                   style={'textAlign': 'center', 'fontSize': 32,
                                          'color':'white','background-color': '#c30045'}),
        #html.Div(html.Img(src = 'assets/background_pic.jpg'),
                 #style={ 'align': 'left'}),
        html.Br(),
        dbc.Row([dbc.Col(html.H4('Credit card holder details'),
                         style={'text-align': 'center','font-size':12, 'color':'white'}, md = 6),
                 dbc.Col(html.H4('Transaction details'),
                         style={'text-align': 'center', 'font-size': 12, 'color': 'white'}, md =6)
                 ]),
        html.Br(),
        dbc.Row([dbc.Col(html.H6('First name:'),
                         style={'text-align': 'right','font-size':12, 'color':'white'},md=2),
                 dbc.Col(dcc.Input(id = 'name',type='text',placeholder='Enter the name',
                                   style= {'textAlign': 'center','width':"230px",
                                           'outline':'none'}, value = ''),
                         style={'textAlign': 'left', 'color':'#415c6c'},md=4),
                dbc.Col(html.H6('Merchant name:'),
                         style={'text-align': 'right', 'font-size': 12, 'color': 'white'}, md=2),
                 dbc.Col(dcc.Input(id='merchant_name', type='text', placeholder='Enter the full name',
                                   style={'textAlign': 'center', 'width': "230px",
                                          'outline': 'none'}),
                         style={'textAlign': 'left', 'color': '#415c6c'}, md=4)
                 ]),
        dbc.Row(html.P()),
        dbc.Row([dbc.Col(html.H6('Last name:'),
                         style={'text-align': 'right','font-size':12, 'color':'white'},md=2),
                 dbc.Col(dcc.Input(id = 'lastname',type='text',placeholder='Enter the name',
                                   style= {'textAlign': 'center','width':"230px",
                                           'outline':'none'}),
                         style={'textAlign': 'left', 'color':'#415c6c'},md=4),
                dbc.Col(html.H6('Amount (USD):'),
                         style={'text-align': 'right', 'font-size': 12, 'color': 'white'}, md=2),
                 dbc.Col(dcc.Input(id='amt', type='text', placeholder='Enter the amount',
                                   style={'textAlign': 'center', 'width': "230px",
                                          'outline': 'none'},value=0),
                         style={'textAlign': 'left', 'color': '#415c6c'}, md=4)
                 ]),
        dbc.Row(html.P()),
        dbc.Row([dbc.Col(html.H6('Date of birth:'),
                         style={'text-align': 'right', 'font-size': 12, 'color': 'white'}, md=2),
                 dbc.Col(dcc.Input(id='dob', type='date', placeholder='',
                                   style={'textAlign': 'center', 'width': "230px",
                                          'outline': 'none'}, value='22-08-1984'),
                         style={'textAlign': 'left', 'color': '#415c6c'}, md=4),
                 dbc.Col(html.H6('Time:'),
                         style={'text-align': 'right', 'font-size': 12, 'color': 'white'}, md=2),
                 dbc.Col(dcc.Input(id='time', type='time', placeholder='', step=1,
                                   style={'textAlign': 'center', 'width': "230px",
                                          'outline': 'none'}, value='00:00:00'),
                         style={'textAlign': 'left', 'color': '#415c6c'}, md=4)
                 ]),
        dbc.Row(html.P()),
        dbc.Row([dbc.Col(html.H6('Profession:'),
                         style={'text-align': 'right', 'font-size': 12, 'color': 'white'}, md=2),
                 dbc.Col(job_dropdown,
                         style={'textAlign': 'left', 'color': '#415c6c','width':"255px" }, md=1),
                 dbc.Col(html.H6('Category:'),
                         style={'text-align': 'right', 'font-size': 12, 'color': 'white','width':'30%'},md=3),
                 dbc.Col(category_dropdown, md=2,
                         style={'textAlign': 'center','width':"255px", 'color':'#415c6c'})
                 ]),
        html.Br(),
        html.Br(),
        dbc.Row([dbc.Col(dbc.Button("Predict",id='button',n_clicks=0,color= 'primary',
                    style={'textAlign': 'center','width':"150px", 'height':'38px' }),
                         style={'textAlign': 'right'} , md = 6),
                 dbc.Col(dbc.Button("Clear", id='reset_button', n_clicks=0, color='secondary',
                                    style={'textAlign': 'center', 'width': "150px", 'height': '38px'}),
                         style={'textAlign': 'left'}, md =6)
                 ]),
        html.Br(),
        html.Br(),
        dbc.Row([dbc.Col(dcc.Textarea(id='predict',style={'whitespace':'pre-line',
                                                            'width':'500px', 'text-align':'center',
                                                          'justify':'center'}, value=''),
                         style={'textAlign': 'center'})]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br()
        ],
        style={'text-align': 'left', 'font-size': 14,
           # Update the background color to the entire app
            'background-color': '#c30045',
           # Change the text color for the whole app
           'color': 'black', 'padding':0, 'margin':0
           },
    fluid=True)

@app.callback(Output('predict','value'),
    Input('button','n_clicks'),
     State('dob', 'value'),
     State('category', 'value'),
     State('job','value'),
     State('time','value'),
     State('amt','value')
     )

def prediction(n_clicks,dob,category,job,time,amt):
    list_names =[[str(dob),str(category),str(job),str(time),amt]]
    predicted_value = apply_model(rf_model,list_names)
    if n_clicks > 0:
        if predicted_value.item() > 0.1:
            return 'The above credit card transaction is fraudulent.'
        else:
            return 'The above credit card transaction is not fraudulent.'

@app.callback(
    [Output('name', 'value'),
    Output('lastname', 'value'),
    Output('dob', 'value'),
    Output('job', 'value'),
     Output('category','value'),
     Output('amt','value'),
     Output('time','value'),
     Output('merchant_name','value')],
    Input('reset_button','n_clicks')
              )

def update_clear(n_clicks):
    if n_clicks > 0:
        return '', '', '', '','','','',''

if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_props_check=False)