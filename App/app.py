import dash
from dash import html, dcc, Input, Output
from model import PostClassifier
import json

model_path = '/Users/alexplash/sentinel/Training/saved_weights.pt'
json_path = '/Users/alexplash/sentinel/Data/testData.json'

# model function definition
def identify_fraudulent_posts(json_path, model_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    posts = [item['POST'] for item in data]
    users = [item['USER'] for item in data]

    classifier = PostClassifier(model_path = '/Users/alexplash/sentinel/Training/saved_weights.pt')
    predictions = classifier.predict(posts)

    fraudulent_users = [users[i] for i, pred in enumerate(predictions) if pred == 1]
    non_fraudulent_users = [users[i] for i, pred in enumerate(predictions) if pred == 0]

    return fraudulent_users, non_fraudulent_users
# end of model function definition

# front end framework
app = dash.Dash(__name__)

app.layout = html.Div(id='main-container', style={
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center',
    'height': '100vh',
}, children=[
    html.Button([
                html.Img(src=app.get_asset_url('database_icon_135716.png'), style={'width': '60px', 'height': '60px'}),
                html.Div('Detect Users', className='button-text'),
                html.Div('- user data uploaded -', style={'color': 'rgba(255, 255, 255, 0.8)', 'fontSize': '10px', 'marginTop': '10px'}),
    ], id='analyze-data-button', className='button'),
    html.Div(id='new-content', style={'display': 'none'})
])

@app.callback(
    Output('main-container', 'children'),
    Input('analyze-data-button', 'n_clicks'),
    prevent_initial_call=True
)
def display_new_content(n_clicks):
    if n_clicks:
        fraudulent_users, non_fraudulent_users = identify_fraudulent_posts(json_path, model_path)

        items = []
        item_container_style = {
            'display': 'flex',
            'flex-direction': 'row',
            'justifyContent': 'space-between',  # Distribute space evenly between user and status
            'alignItems': 'center',
            'margin': '10px',
            'padding': '0 20px',  # Added padding for each row
            'border': '2px solid white',
            'border-radius': '10px',
            'background-color': 'rgba(0, 0, 0, 0.8)',  # Slight transparency for row background
        }
        user_style = {
            'flex-grow': '1',  # User box grows to fill space, ensuring alignment
            'margin': '5px',
            'padding': '5px 10px',
            'border-radius': '10px',
            'border': '2px solid white',
        }
        status_style = lambda color: {
            'flex-basis': '150px',  # Fixed basis for the status box
            'text-align': 'center',
            'border-radius': '10px',
            'border': '2px solid white',
            'padding': '5px 10px',
            'margin': '5px',
            'background-color': color,
            'color': 'white'
        }

        for user in non_fraudulent_users:
            items.append(html.Div([
                html.Div(user, style=user_style),
                html.Div('Unlikely Fraudulent', style=status_style('green'))
            ], style=item_container_style))

        for user in fraudulent_users:
            items.append(html.Div([
                html.Div(user, style=user_style),
                html.Div('Likely Fraudulent', style=status_style('red'))
            ], style=item_container_style))

        content = html.Div(style={
            'overflow-y': 'scroll',
            'height': '75vh',  # Fill most of the container's vertical space
            'width': '90%',  # Width adjusted to fill most of the container's horizontal space
            'max-width': 'none',  # Remove max-width constraint
        }, children=items)

        return dcc.Loading(
            type="circle",
            children=html.Div(style={
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
                'width': '100%',
                'height': '100%',
            }, children=content)
        )

if __name__ == '__main__':
    app.run_server(debug=False)
