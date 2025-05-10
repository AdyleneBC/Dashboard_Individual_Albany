#Creamos el archivo de la APP en el interprete principal (Phyton)

#####################################################
#Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import seaborn as sns 

#Cragamos librerias
import numpy as np 
import scipy.special as special
from scipy.optimize import curve_fit

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from streamlit_option_menu import option_menu
#from numerize.numerize import numerize



st.set_page_config(page_title="Dashboard Albany", page_icon="🌎", layout="wide")

######################################################
#Definimos la instancia
@st.cache_resource

######################################################
#Creamos la función de carga de datos
def load_data():
   #Lectura del archivo csv
   df=pd.read_csv("Albany.csv", index_col= 'host_id')

   #Selecciono las columnas tipo numericas del dataframe
   numeric_df = df.select_dtypes(['float','int'])  #Devuelve Columnas
   numeric_cols= numeric_df.columns                #Devuelve lista de Columnas

   #Selecciono las columnas tipo texto del dataframe
   text_df = df.select_dtypes(['object'])  #Devuelve Columnas
   text_cols= text_df.columns              #Devuelve lista de Columnas
   
   #Selecciono algunas columnas categoricas de valores para desplegar en diferentes cuadros
   categorical_column_verifications= df['host_verifications']
   #Obtengo los valores unicos de la columna categórica seleccionada
   unique_categories_verifications= categorical_column_verifications.unique()

   return df, numeric_cols, text_cols, unique_categories_verifications, numeric_df


###############################################################################
#Cargo los datos obtenidos de la función "load_data"
df, numeric_cols, text_cols, unique_categories_verifications, numeric_df = load_data()

###############################################################################
#############CREACIÓN DEL DASHBOARD##################

#NAVEGADOR


#SIDEBAR
st.sidebar.image("logo_albany.png", caption="Albany")


# Fondo con color sólido
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, 
             rgb(136, 14, 79),   /* rosa fuerte */
            rgb(91, 44, 111),    /* morado */
            rgb(20, 20, 20)       /* casi negro */
        );
        background-attachment: fixed;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

##############################################################################
# Encabezados del sidebar
#st.sidebar.title("DASHBOARD ALBANY")
# Logo en el sidebar
#st.sidebar.image("logo.png", width=150)
#st.sidebar.header("⚙️ Sidebar")
#st.sidebar.subheader("📁 Panel de selección")

# Inicializar estado si no existe
if "submenu" not in st.session_state:
    st.session_state.submenu = "Albany"  # por default

# Botones del submenú
if st.sidebar.button("Albany", use_container_width=True):
    st.session_state.submenu = "Albany"


if st.sidebar.button("Ver Gráficas", use_container_width=True):
    st.session_state.submenu = "Gráficas"

if st.sidebar.button("Regresión", use_container_width=True):
    st.session_state.submenu = "Regresión"


    



if st.session_state.submenu == "Albany":
    st.title("Albany, NY, USA")
    # Texto descriptivo
    st.markdown("""
    **Albany** es la capital del estado de Nueva York y del condado de Albany, en los Estados Unidos. 
    Tiene una población de 97 750 habitantes en 2010.
    """)

    # Imagen desde archivo local


    st.markdown("### Conoce los principales monumentos")

    image_urls = [
        "https://www.redfin.com/blog/wp-content/uploads/2023/07/Albany-NY-skyline.jpg",
        "https://www.momondo.mx/rimg/dimg/c8/20/30272686-city-5657-1683e7d483a.jpg?width=1366&height=768&xhint=2133&yhint=1864&crop=true",
        "https://assets.hvmag.com/2023/05/albany-AdobeStock_153544781-1068x712.jpg", 
        "https://res.cloudinary.com/sv-albany/image/upload/v1744759211/cms_resources/clients/albany/Architecture_c20e3566-0a06-4fe5-a0af-9090ab647c75.jpg"
    ]

    carousel_html = f"""
    <style>
    .carousel-container {{
        position: relative;
        width: 100%;
        max-width: 700px;
        margin: auto;
        overflow: hidden;
        border-radius: 12px;
        box-shadow: 0 0 20px pink;
    }}

    .carousel-slide {{
        display: block;
        height: 100%;
    }}

    .carousel-slide img {{
        display: none;
        width: 100%;
        border-radius: 12px;
    }}

    .carousel-slide img.active {{
        display: block;
    }}

    .carousel-button {{
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background-color: rgba(0, 0, 0, 0.5);
        border: none;
        color: white;
        font-size: 2rem;
        padding: 8px 16px;
        cursor: pointer;
        z-index: 1;
        border-radius: 50%;
    }}

    #prevBtn {{
        left: 10px;
    }}

    #nextBtn {{
        right: 10px;
    }}
    </style>

    <div class="carousel-container">
        <button class="carousel-button" id="prevBtn">&#10094;</button>
        <div class="carousel-slide" id="carousel">
            {''.join([f'<img src="{img}" class="{ "active" if i == 0 else "" }" />' for i, img in enumerate(image_urls)])}
        </div>
        <button class="carousel-button" id="nextBtn">&#10095;</button>
    </div>

    <script>
    let slideIndex = 0;
    const slides = document.querySelectorAll('#carousel img');
    const totalSlides = slides.length;
    const intervalTime = 3000;
    let autoSlide = setInterval(showNextSlide, intervalTime);

    function updateSlide() {{
        slides.forEach((img, i) => {{
            img.classList.remove('active');
            if (i === slideIndex) {{
                img.classList.add('active');
            }}
        }});
    }}

    function showNextSlide() {{
        slideIndex = (slideIndex + 1) % totalSlides;
        updateSlide();
    }}

    function showPrevSlide() {{
        slideIndex = (slideIndex - 1 + totalSlides) % totalSlides;
        updateSlide();
    }}

    document.getElementById('nextBtn').addEventListener('click', () => {{
        clearInterval(autoSlide);
        showNextSlide();
        autoSlide = setInterval(showNextSlide, intervalTime);
    }});

    document.getElementById('prevBtn').addEventListener('click', () => {{
        clearInterval(autoSlide);
        showPrevSlide();
        autoSlide = setInterval(showNextSlide, intervalTime);
    }});
    </script>
    """

    st.components.v1.html(carousel_html, height=450)



    # Video desde YouTube
    st.markdown("### Sumergete en la Ciudad de Albany")
    st.video("https://www.youtube.com/watch?v=sOYly0of8Js")





    st.header("Base de datos de Albany")

        # Selección de columnas
    selected_columns = st.multiselect(
        "Selecciona las columnas que deseas visualizar:",
        df.columns.tolist(),  # Lista de todas las columnas
        default=df.columns[:5]  # Mostrar por defecto las primeras 5 columnas
    )

    # Mostrar solo las columnas seleccionadas
    if selected_columns:
        st.dataframe(df[selected_columns])
    else:
        st.warning("⚠️ Por favor selecciona al menos una columna para visualizar.")

    # Checkbox para mostrar la tabla
    if st.checkbox("Mostrar tabla completa"):
        st.dataframe(df)

    neighborhoods = df['neighbourhood'].unique()
    selected_neigh = st.multiselect("Filtrar por vecindario", neighborhoods, default=neighborhoods[:1])
    room_types = df['room_type'].unique()
    selected_room = st.selectbox("Tipo de habitación", room_types)

    # Crear dos columnas
    colf1, colf2 = st.columns(2)
    with colf1:
        filtered_df = df[df['neighbourhood'].isin(selected_neigh)]
    with colf2:
        filtered_df = filtered_df[filtered_df['room_type'] == selected_room]

    # Crear dos columnas
    col1, col2, col3, col4 = st.columns(4)
    # Mostrar las métricas en cada columna
    with col1:
        st.metric("Total de registros", len(filtered_df))
    with col2:
        st.metric("Precio promedio", f"${filtered_df['price'].mean():.2f}")
    with col3:
        st.metric("Precio mínimo", f"${filtered_df['price'].min():.2f}")
    with col4:
        st.metric("Precio máximo", f"${filtered_df['price'].max():.2f}")


    max_price_property = filtered_df.loc[filtered_df['price'].idxmax()]
    min_price_property = filtered_df.loc[filtered_df['price'].idxmin()]

    st.write("Propiedad más cara:")
    st.write(max_price_property[['name', 'price']])

    st.write("Propiedad más barata:")
    st.write(min_price_property[['name', 'price']])






elif st.session_state.submenu == "Gráficas":
    st.header("Gráficas de Albany")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_registros = len(df)
        st.metric(label="Total de registros", value=total_registros, delta="+120")
    with col2:
        total_columnas = df.shape[1]
        st.metric(label="Total de campos", value=total_columnas, delta="-120")
    with col3:
        price = df['price'].median()
        st.metric(label="Precio promedio", value=price, delta="-120")
    with col4:
        st.metric("Disponibilidad promedio", round(df["availability_365"].mean(), 1), delta = "+120")


    View= st.sidebar.selectbox(label= "Tipo de gráfica", options= ["Barras", "Pastel", "Dispersión"])

    if View == "Barras":
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x='price', title='Distribución de precios', nbins=50, text_auto=True, color_discrete_sequence=['#b61a6d'])
            fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            st.plotly_chart(fig)

        with col2:
            fig = px.box(df, x='room_type', y='price', title='Precio por tipo de habitación', color='room_type')
            st.plotly_chart(fig)

        ######
        avg_price_by_property = df.groupby('property_type')['price'].mean().sort_values(ascending=False)
        avg_price_df = avg_price_by_property.reset_index()

        fig = px.bar(avg_price_df, 
                    x='property_type', 
                    y='price', 
                    title='Precio promedio por tipo de propiedad',
                    labels={'price': 'Precio Promedio', 'property_type': 'Tipo de Propiedad'}, color_discrete_sequence=['#b61a6d'])  
        st.plotly_chart(fig)

        #####
        col1, col2 = st.columns(2)

        with col1:
            host_verification_counts = df['host_verifications'].value_counts()
            fig = px.bar(host_verification_counts, x=host_verification_counts.index, y=host_verification_counts.values, title='Número de propiedades por verificación de host', color_discrete_sequence=['#b61a6d'])
            st.plotly_chart(fig)

        ####
        with col2:
            # Calcula los conteos
            host_verification_counts = df['host_verifications'].value_counts()
            # Convierte los datos a un DataFrame
            tabla_verificaciones = pd.DataFrame({
                'Verificación': host_verification_counts.index,
                'Número de propiedades': host_verification_counts.values
            })
            # Título
            ##st.subheader('Número de propiedades por verificación de host')
            st.markdown("<h3 style='text-align: center;'>Número de propiedades por verificación de host</h3>", unsafe_allow_html=True)
            # Muestra la tabla
            ##st.table(tabla_verificaciones)
            ##st.dataframe(tabla_verificaciones.style.background_gradient(cmap='Blues'))
            st.dataframe(
                tabla_verificaciones.style.set_properties(**{
                    'background-color': 'rgba( 136, 14, 79 , 0.7)',  # blanco con 70% opacidad
                    'color': 'white',
                    'border-color': 'gray',
                })
            )




        #Generamos los encabezados para el dashboard
        st.title("Bar Plot Interactivo")
        #st.header("Panel Principal")
        #st.subheader("Bar Plot")
        
        #Menus desplegables de opciones de la variables seleccionadas
        # Definición manual de columnas categóricas y numéricas
        variables_categoricas = ['room_type', 'neighbourhood_cleansed', 'property_type', 'host_is_superhost']  # ← tú decides
        variables_numericas = ['price', 'accommodates', 'bedrooms', 'number_of_reviews', 'review_scores_rating']

        Variable_cat= st.sidebar.selectbox(label= "Variable Categórica", options= variables_categoricas)
        Variable_num= st.sidebar.selectbox(label= "Variable Numérica", options= variables_numericas)
        # Menú desplegable para elegir la variable categórica para el color

        
    #GRAPH 4: BARPLOT
    #Despliegue de un bar plot, definiendo las variables "X categorias" y "Y numéricas" 
        figure4 = px.bar(data_frame=df, x=Variable_cat, y=Variable_num,
                 title=f'Distribución de {Variable_num} por {Variable_cat}',  color=Variable_cat)
        figure4.update_xaxes(automargin=True)
        figure4.update_yaxes(automargin=True)
        st.plotly_chart(figure4)

    elif View == "Pastel":
        # Definición manual de columnas categóricas y numéricas
        variables_categoricasP = ['room_type', 'neighbourhood_cleansed', 'property_type', 'host_is_superhost']  # ← tú decides
        variables_numericasP = ['price', 'accommodates', 'bedrooms', 'number_of_reviews', 'review_scores_rating']
            #Menus desplegables de opciones de la variables seleccionadas
        Variable_cat= st.sidebar.selectbox(label= "Variable Categórica", options= variables_categoricasP )
        Variable_num= st.sidebar.selectbox(label= "Variable Numérica", options= variables_numericasP)

        tipo_grafica = st.sidebar.radio("Tipo de gráfica", options=["Pastel", "Dona"])

    #GRAPH 3: PIEPLOT
    #Despliegue de un pie plot, definiendo las variables "X categorias" y "Y numéricas" 
        col1, col2 = st.columns([2, 1])
        with col1:
            figure3 = px.pie(data_frame=df, names=df[Variable_cat], 
                        values= df[Variable_num], title=f'Distribución de {Variable_num} por {Variable_cat}', hole=0.4 if tipo_grafica == "Dona" else 0,  # Si es dona, crea el agujero
                        width=800, height=600)
            st.plotly_chart(figure3)

        with col2:
            grupo = df.groupby(Variable_cat)[Variable_num].sum()
            total = grupo.sum()
            porcentajes = (grupo / total * 100).round(2)

            # Ordenar de mayor a menor
            porcentajes = porcentajes.sort_values(ascending=False)

            st.markdown("### Porcentajes por categoría")
            for cat, valor in porcentajes.items():
                color = "green" if valor >= 50 else "red"
                st.metric(label=cat, value=f"{valor}%", delta=None)
                st.markdown(f"<div style='color:{color}; font-weight:bold'>{valor}%</div>", unsafe_allow_html=True)

        
        # Mostrar porcentajes directamente sobre las secciones
        #figure3.update_traces(
         #   textinfo='label+percent',     # ← esto muestra etiqueta + porcentaje
          #  textposition='inside',        # ← para que aparezcan dentro del gráfico
           # hoverinfo='label+value+percent'  # ← info al pasar el mouse
        #)
            
                # Calcular porcentaje de superhosts
       

        f8 = px.histogram(df, x=Variable_num, color=Variable_cat, title='Histograma')
        st.plotly_chart(f8)

        f9 = px.treemap(df, path=[Variable_cat], values=Variable_num, color_continuous_scale='Reds')
        f9.update_layout(
            title='Distribución de propiedades por categoría',
            width=1000,  # Ancho
            height=400  # Alto
        )

        st.plotly_chart(f9)



    elif View == 'Dispersión':
        # Crear el gráfico de mapa coroplético
        fig = px.choropleth(
            df,
            locations='host_location',  # Columna con nombres de países
            locationmode='country names',  # Usa nombres en lugar de códigos ISO
            color='availability_365',  # Valor numérico para colorear los países
            color_continuous_scale='Blues',  # Escala de colores
            title='Número de propiedades disponibles por país',
            labels={'availability_365': 'Propiedades'},  # Etiqueta personalizada
            width=900,
            height=600
        )

        # Mostrar el mapa en Streamlit
        st.plotly_chart(fig)

        x_selected= st.sidebar.selectbox(label= "x", options= numeric_cols)
        y_selected= st.sidebar.selectbox(label= "y", options= numeric_cols)

        col1, col2 = st.columns(2)
        with col1:
            figure2 = px.scatter(data_frame=numeric_df, x=x_selected, y= y_selected, title= 'Dispersiones')
            st.plotly_chart(figure2)
        
        with col2:
            fig = px.line(df, x='accommodates', y='beds',color='host_is_superhost', title='Húespedes')
            fig.update_traces(mode='lines+markers')  # Opcional: muestra líneas y puntos
            st.plotly_chart(fig)



        fig2 = px.scatter(
            df,
            x='beds',
            y='accommodates',
            size='availability_365',  # Tamaño de burbuja según área
            hover_name='price',  # Muestra el nombre al pasar el mouse
            color='price',
            title='beds vs. accommodates, teniendo el precio'
        )
        st.plotly_chart(fig2)

elif st.session_state.submenu == "Regresión":
    st.header("Regresiones de Albany")

    # Suponemos que ya tienes cargado un DataFrame llamado df
    # Puedes modificar estas listas según tus columnas numéricas
    variables_numericas = ['price', 'accommodates', 'bedrooms', 'number_of_reviews', 'review_scores_rating']

    # Menú para seleccionar el tipo de regresión
    tipo_regresion = st.sidebar.radio("Selecciona tipo de regresión:", ["Lineal simple", "Lineal múltiple", "Regresión Logística"])

    # Seleccionar variable dependiente (y)
    target = st.sidebar.selectbox("Variable objetivo (Y)", variables_numericas)

    # Regresión lineal simple
    if tipo_regresion == "Lineal simple":
        st.header("Regresión lineal simple")
        # Seleccionar solo una variable independiente (x)
        feature = st.sidebar.selectbox("Variable predictora (X)", [v for v in variables_numericas if v != target])

        # Eliminar valores nulos
        df_filtrado = df[[feature, target]].dropna()

        X = df_filtrado[[feature]]
        y = df_filtrado[target]

        modelo = LinearRegression()
        modelo.fit(X, y)

        df_filtrado['predicciones'] = modelo.predict(X)

        # Gráfico
        fig = px.scatter(df_filtrado, x=feature, y=target, title=f'Regresión Lineal Simple: {target} vs {feature}')
        fig.add_traces(px.line(df_filtrado, x=feature, y='predicciones').data)
        st.plotly_chart(fig)

        # Mostrar métricas del modelo
        st.markdown(f"**Ecuación:**  \n{target} = {modelo.coef_[0]:.3f} × {feature} + {modelo.intercept_:.3f}")

    # Regresión lineal múltiple
    elif tipo_regresion == "Lineal múltiple":
        st.header("Regresión lineal múltiple")
        # Selección de múltiples predictores
        features = st.sidebar.multiselect("Variables predictoras (X)", [v for v in variables_numericas if v != target])

        if features:
            df_filtrado = df[features + [target]].dropna()

            X = df_filtrado[features]
            y = df_filtrado[target]

            modelo = LinearRegression()
            modelo.fit(X, y)

            y_pred = modelo.predict(X)

            # Mostrar resultados
            st.write("**Coeficientes:**")
            for f, coef in zip(features, modelo.coef_):
                st.write(f"- {f}: {coef:.3f}")
            st.write(f"**Intercepto:** {modelo.intercept_:.3f}")

            # Mostrar gráfico comparando valores reales vs predichos
            resultado_df = pd.DataFrame({'Real': y, 'Predicho': y_pred})
            fig = px.scatter(resultado_df, x='Real', y='Predicho', trendline='ols',
                            title='Comparación de valores reales vs predichos')
            st.plotly_chart(fig)

    elif tipo_regresion == "Regresión Logística":
        st.header("Regresión Logística")

        df['host_is_superhost'] = df['host_is_superhost'].replace([1, 2, 2.9], 0).replace([3, 4, 5, 6, 7], 1)
        df['host_identity_verified'] = df['host_identity_verified'].replace(['f'], 0).replace(['t'], 1)

        variables_numericas = ['price', 'accommodates', 'bedrooms', 'number_of_reviews', 'review_scores_rating']
        variables_binarias = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']  # Agrega aquí más columnas binarias

        # Asegúrate de convertir todas las binarias a 0 y 1 si no lo están
        df['host_is_superhost'] = df['host_is_superhost'].replace([1, 2, 2.9], 0).replace([3, 4, 5, 6, 7], 1)
        df['host_has_profile_pic'] = df['host_has_profile_pic'].apply(lambda x: 1 if x in ['t', 'yes', 'True', 1] else 0)
        df['host_identity_verified'] = df['host_identity_verified'].apply(lambda x: 1 if x in ['t', 'yes', 'True', 1] else 0)

        # Selección de variable binaria (solo una)
        target = st.sidebar.selectbox("Variable Objetivo (Binaria)", options=variables_binarias)

        # Selección de variables predictoras
        features = st.sidebar.multiselect("Variables Predictoras", options=[v for v in variables_numericas if v != target])


        if target and features:
            df_filtrado = df[features + [target]].dropna()

            # Convertir target a binario si es texto
            if df_filtrado[target].dtype == 'object':
                df_filtrado[target] = df_filtrado[target].apply(lambda x: 1 if x in ['t', 'yes', 'True', '1'] else 0)

            X = df_filtrado[features]
            y = df_filtrado[target]

            # Dividir en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Crear y entrenar modelo
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import classification_report
            modelo = LogisticRegression()
            modelo.fit(X_train, y_train)

            # Predicciones
            y_pred = modelo.predict(X_test)

            # Mostrar resultados
            st.subheader("Métricas del Modelo")
            st.text("Matriz de Confusión:")
            st.write(confusion_matrix(y_test, y_pred))

            st.text("Reporte de Clasificación:")
            st.text(classification_report(y_test, y_pred))

            # Mostrar gráfico
            resultado_df = pd.DataFrame({'Real': y_test, 'Predicho': y_pred})
            fig = px.histogram(resultado_df, x='Predicho', color='Real',
                            title='Distribución de Predicciones vs Reales')
            st.plotly_chart(fig)







    
