import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# --- Leitura e limpeza dos dados ---
df = pd.read_csv("ecommerce_estatistica.csv")
df['Nota'] = pd.to_numeric(df['Nota'], errors='coerce')
df = df.dropna(subset=['Nota'])

# --- Gráficos ---
# Histograma
fig_hist = px.histogram(df, x='Nota', nbins=10, title='Distribuição das Notas dos Produtos')
fig_hist.update_layout(xaxis_title='Nota', yaxis_title='Frequência')

# Dispersão
fig_disp = px.scatter(df, x='Preço_MinMax', y='Nota', color='Desconto',
                      title='Preço (Normalizado) vs Nota',
                      labels={"Preço_MinMax": "Preço Normalizado", "Nota": "Nota"})

# Mapa de calor de correlação
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr().round(2)
fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu',
                     title="Mapa de Calor das Correlações")

# Barras: Top 10 marcas
top_marcas = df.groupby('Marca')['Qtd_Vendidos_Cod'].sum().sort_values(ascending=False).head(10)
fig_bar = px.bar(x=top_marcas.values, y=top_marcas.index, orientation='h',
                 title='Top 10 Marcas com Mais Vendas',
                 labels={'x': 'Quantidade Vendida (Codificada)', 'y': 'Marca'})

# Pizza: Gênero
genero_counts = df['Gênero'].value_counts()
fig_pizza = px.pie(values=genero_counts.values, names=genero_counts.index,
                   title='Distribuição dos Produtos por Gênero')

# Densidade
fig_dens = px.density_contour(df, x='Nota', title='Densidade das Notas dos Produtos')

# Regressão: Desconto vs Vendas
fig_reg = px.scatter(df, x='Desconto', y='Qtd_Vendidos_Cod', trendline='ols',
                     title='Desconto vs Quantidade Vendida (Codificada)',
                     labels={'Desconto': 'Desconto (%)', 'Qtd_Vendidos_Cod': 'Qtd. Vendida'})

# --- Layout da Aplicação ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Visualização E-commerce"

app.layout = dbc.Container([
    html.H1("Dashboard de Estatísticas - E-commerce", className="text-center my-4"),
    dcc.Tabs([
        dcc.Tab(label='Histograma', children=[dcc.Graph(figure=fig_hist)]),
        dcc.Tab(label='Dispersão', children=[dcc.Graph(figure=fig_disp)]),
        dcc.Tab(label='Mapa de Calor', children=[dcc.Graph(figure=fig_heat)]),
        dcc.Tab(label='Barras', children=[dcc.Graph(figure=fig_bar)]),
        dcc.Tab(label='Pizza', children=[dcc.Graph(figure=fig_pizza)]),
        dcc.Tab(label='Densidade', children=[dcc.Graph(figure=fig_dens)]),
        dcc.Tab(label='Regressão', children=[dcc.Graph(figure=fig_reg)]),
    ])
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)
