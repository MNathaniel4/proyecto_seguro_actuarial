

import altair as alt
import plotly.express as px

def histograma_altair(data, variable_num, variable_cat, max_bins = 25, discriminar = True):
    histograma = alt.Chart(data).mark_bar().encode(
        alt.X(f'{variable_num}', bin = alt.Bin(maxbins = max_bins),
              title = variable_num.replace('_', ' ').title()),
        alt.Y('count():Q', title = ' Numero de Pólizas'),
        tooltip = [f'{variable_num}:Q', 'count():Q']
        )
    
    if discriminar:
        grafico_final = histograma.facet(
            facet = alt.Facet(f'{variable_cat}:N', title = variable_cat.replace('_', ' ').title()),
            columns = 4
        ).interactive()
    else:
        grafico_final = histograma
    return grafico_final


def histograma_plotly(data, variable_num, variable_cat, nbins = 40, por_cateogria = True):
    # Generamos el histograma con el marginal tipo box
    if por_cateogria:
        fig = px.histogram(
            data, 
            x=variable_num, 
            color=variable_cat,           
            marginal="box",               
            barmode="overlay",            
            nbins= nbins,                     
            title=f"Distribución de {variable_num.replace('_', ' ').title()} por {variable_cat.replace('_', ' ').title()}",
            template="plotly_white",      # Un fondo blanco limpio y estético
            color_discrete_sequence=px.colors.qualitative.Safe # Paleta de colores moderna
        )
    else: 
         fig = px.histogram(
            data, 
            x=variable_num, 
         
            marginal="box",               
            barmode="overlay",            
            nbins= nbins,                     
            title=f"Distribución de {variable_num.replace('_', ' ').title()}",
            template="plotly_white",      # Un fondo blanco limpio y estético
            color_discrete_sequence=px.colors.qualitative.Safe # Paleta de colores moderna       
        )
         
    fig.update_layout(
    height=550,

    title_font_size=18,
    hovermode="x unified"         # Agrupa el tooltip para ver todas las categorías juntas al pasar el mouse
    )
    fig.show()