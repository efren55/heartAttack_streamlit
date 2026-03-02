import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json

st.set_page_config(page_title="Heart Attack Analysis Dashboard", layout="wide") #page_title para poner un título a la página, layout="wide" para que la página ocupe todo el ancho de la pantalla

#Guardar en cache la respuesta del llm, para esto aplicar un if si la respuesta ya esta guardada en cache, entonces se muestra la respuesta guardada, si no, se hace la consulta al llm y se guarda la respuesta en cache para futuras consultas.
@st.cache_data(show_spinner=False) #show_spinner=False para no mostrar el spinner de carga cada vez que se consulta al llm, ya que la respuesta se guarda en cache y se muestra inmediatamente en futuras consultas.
def guardar_respuesta(respuesta):
    
    os.makedirs("temp", exist_ok=True)#existe la carpeta temp, si no existe se crea
    ruta_archivo = os.path.join("temp", "cache.json")
    with open(ruta_archivo,"w", encoding="utf-8") as f: #w para escribir el archivo, encoding="utf-8" para evitar problemas con caracteres especiales
        # Si 'respuesta' es el objeto de LangChain, guardamos solo el contenido (.content)
        json.dump({"contenido": respuesta.content}, f, ensure_ascii=False, indent=4) #dump para guardar el contenido en formato json, ensure_ascii=False para permitir caracteres especiales, indent=4 para que el json se vea bonito, f para indicar el archivo donde se guarda la respuesta


api_key=os.environ.get("GOOGLE_API_KEY")
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.5)


def main(): 
    st.title("Heart Attack Analysis Dashboard")
    df=pd.read_csv("./Medicaldataset.csv")
    df_dict=df.to_dict(orient="records")
    ruta_cache = os.path.join("temp", "cache.json")

    st.subheader("Heart Attack Table:")
    st.dataframe(df)
    st.subheader("Using LLM for analizing the dataset:")
    if st.button("Analyze Dataset with LLM"):
        prompt=ChatPromptTemplate.from_messages([
            ("system","You are a data science expert and offer recommendations on how to analyze the dataset. You also indicate the main variables to analyze and the best charts and machine learning models to predict the dataset."),
            ("user","Analyze the following dataset and share your opinion, recommendations, and what it's about. Here is the dataset: {dataset}")
        ])
        chain = prompt | llm
        with st.spinner("Generating analysis..."): #spinner para mostrar que se está generando el análisis, mientras se espera la respuesta del llm. with para mostrar el spinner solo durante la generación del análisis, una vez que se obtiene la respuesta, se muestra en pantalla y se guarda en cache, y el spinner desaparece automáticamente.
            response = chain.invoke({"dataset": df_dict})
            
            # 1. Mostrar en pantalla
            st.markdown(response.content)
            
            # 2. Guardar físicamente
            guardar_respuesta(response)
            st.success("Answer saved in cache for future use")
    
    else:
        # Si NO se presionó el botón, intentamos cargar lo que haya en cache
        if os.path.exists(ruta_cache):
            with open(ruta_cache, "r", encoding="utf-8") as f: #r para leer el archivo, encoding="utf-8" para evitar problemas con caracteres especiales
                datos_viejos = json.load(f)
                st.info("Showing cached analysis from previous LLM response")
                st.markdown(datos_viejos["contenido"])

    st.subheader("Exploratory Data Analysis (EDA):")
    st.write("Null values:")
    st.write(df.isnull().sum())
    st.write("General information of the dataset:")
    st.write(df.describe())
    st.write("We are noticed that there are some outlier values in the area of 'Heart rate' that are a higher that expected. A normal person has a heart rate between 60 and 100 bpm, but in the dataset we can see that there is a value of 1111 bpm and that is psichologically impossible, so we can consider that value as an outlier and we can remove it from the dataset, in fact people can't have a 300 bpm heart rate, so we can consider that value as an outlier and we can remove it from the dataset.")
    df.loc[df["Heart rate"]>300] = df["Heart rate"].mean()
    st.write(df.describe())
    with st.container():
        col1,col2,col3=st.columns(3)
        with col1:
            st.write("Result of the dataset:")
            st.write(df["Result"].value_counts())
        with col2:
            st.write("Age")
            st.write(df["Age"].value_counts().head(5))
        with col3:
            st.write("Gender")
            st.write(df["Gender"].value_counts())
    
    with st.container():
        col1,col2,col3=st.columns(3)
        with col1:
            st.write("Heart rate")
            st.write(df["Heart rate"].value_counts().head(5))
        with col2:
            st.write("Systolic blood pressure")
            st.write(df["Systolic blood pressure"].value_counts().head(5))
        with col3:
            st.write("Diastolic blood pressure")
            st.write(df["Diastolic blood pressure"].value_counts().head(5))
    with st.container():
        col1,col2,col3=st.columns(3)
        with col1:
            st.write("Blood sugar")
            st.write(df["Blood sugar"].value_counts().head(5))
        with col2:
            st.write("CK-MB")
            st.write(df["CK-MB"].value_counts().head(5))
        with col3:
            st.write("Troponin")
            st.write(df["Troponin"].value_counts().head(5))

    st.write("Deleting numerical values in the 'Result' column and keeping only 'positive' and 'negative' values, because we want to analyze the dataset with a binary classification model, and we need to have only two classes in the target variable, so we can consider that the numerical values in the 'Result' column are not useful for our analysis, and we can remove them from the dataset.")
    df = df[df["Result"].isin(["positive","negative"])]
    st.write(df["Result"].value_counts())

    st.subheader("Graphical EDA:")
    with st.container():
        col1,col2,col3=st.columns(3)
        with col1:
            fig1=px.histogram(df,x="Age",nbins=20,title="Age distribution")
            fig1.update_traces(marker=dict(line=dict(color="black",width=1)))
            st.plotly_chart(fig1)
        with col2:
            result=df["Result"].value_counts()
            fig2=px.bar(result,x=result.index.astype("str"),y=result.values,title="Result distribution",color=result.index.astype("str"))
            st.plotly_chart(fig2)
        with col3:
            fig3=px.bar(df["Gender"].value_counts(),x=df["Gender"].value_counts().index.astype("str"),y=df["Gender"].value_counts().values,title="Gender distribution",color=df["Gender"].value_counts().index.astype("str"))
            fig3.update_xaxes(
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Female", "Male"]
            )

            st.plotly_chart(fig3)
    with st.container():
        col1,col2,col3=st.columns(3)
        with col1:
            fig4=px.histogram(df,x="Heart rate",nbins=20,title="Heart rate distribution")
            fig4.update_traces(marker=dict(line=dict(color="black",width=1)))
            st.plotly_chart(fig4)
        with col2:
            fig5=px.histogram(df,x="Systolic blood pressure",nbins=20,title="Systolic blood pressure distribution")
            fig5.update_traces(marker=dict(line=dict(color="black",width=1)))
            st.plotly_chart(fig5)
        with col3:
            fig6=px.histogram(df,x="Diastolic blood pressure",nbins=20,title="Diastolic blood pressure distribution")
            fig6.update_traces(marker=dict(line=dict(color="black",width=1)))
            st.plotly_chart(fig6)
    with st.container():
        col1,col2,col3=st.columns(3)
        with col1:
            fig7=px.histogram(df,x="Blood sugar",nbins=20,title="Blood sugar distribution")
            fig7.update_traces(marker=dict(line=dict(color="black",width=1)))
            st.plotly_chart(fig7)
        with col2:
            fig8=px.histogram(df,x="CK-MB",nbins=20,title="CK-MB distribution")
            fig8.update_traces(marker=dict(line=dict(color="black",width=1)))
            st.plotly_chart(fig8)
        with col3:
            fig9=px.histogram(df,x="Troponin",nbins=20,title="Troponin distribution")
            fig9.update_traces(marker=dict(line=dict(color="black",width=1)))
            st.plotly_chart(fig9)

    st.header("Multivariable Data Analysis:")
    st.subheader("Relationship between Age and Heart rate:")
    df_pair=df.loc[:,["Age","Heart rate","Result"]]
    g= sns.pairplot(df_pair,hue="Result",height=2.5,aspect=1.8)
    st.pyplot(g)
    st.subheader("Relationship between Result and the other variables:")
    with st.container():
        col1,col2,col3,col4=st.columns(4)
        with col1:
            fig10=px.box(df,x="Result",y="Age",color="Result",title="Result vs Age")
            st.plotly_chart(fig10)
        with col2:
            fig11=px.box(df,x="Result",y="Heart rate",color="Result",title="Result vs Heart rate")
            st.plotly_chart(fig11)
        with col3:
            fig12=px.box(df,x="Result",y="Systolic blood pressure",color="Result",title="Result vs Systolic blood pressure")
            st.plotly_chart(fig12)
        with col4:
            fig13=px.box(df,x="Result",y="Diastolic blood pressure",color="Result",title="Result vs Diastolic blood pressure")
            st.plotly_chart(fig13)
    with st.container():
        col1,col2,col3=st.columns(3)
        with col1:
            fig14=px.box(df,x="Result",y="Blood sugar",color="Result",title="Result vs Blood sugar")
            st.plotly_chart(fig14)
        with col2:
            fig15=px.box(df,x="Result",y="CK-MB",color="Result",title="Result vs CK-MB")
            fig15.update_yaxes(type="log")
            st.plotly_chart(fig15)
        with col3:
            fig16=px.box(df,x="Result",y="Troponin",color="Result",title="Result vs Troponin")
            fig16.update_yaxes(type="log")
            st.plotly_chart(fig16)

    fig17=px.histogram(df,x="Gender",color="Result",barmode="group",title="Gender vs Result",labels={"Gender":"Gender","Result":"Result"}  )
    fig17.update_xaxes(
    tickmode="array",
    tickvals=[0, 1],
    ticktext=["Female", "Male"])
    st.plotly_chart(fig17)

    st.header("Using random forest classifier to predict the dataset:")
    if 'df_importance_dict' not in st.session_state:
        st.session_state.df_importance_dict = None

    if st.button("Train Random Forest Classifier"):
        with st.spinner("Generating analysis..."):
            st.subheader("Accuracy of the model:")
            features=["Age","Heart rate","Systolic blood pressure","Diastolic blood pressure","Blood sugar","CK-MB","Troponin"]
            X=df[features]
            y=df["Result"]
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            rf_model=RandomForestClassifier(n_estimators=200,max_depth=None,random_state=42)
            rf_model.fit(X_train,y_train)
            y_pred=rf_model.predict(X_test)
            st.write(f"Accuracy of the model:{accuracy_score(y_test,y_pred)}")
            st.subheader("Feature importance:")
            X_columnas=X_train.columns
            importancia_var=rf_model.feature_importances_
            #df_importance=pd.DataFrame({"Variables":X_columnas,"Importance":importancia_var})
            #df_importance=df_importance.sort_values(by="Importance",ascending=False)
            df_imp = pd.DataFrame({"Variables": X_train.columns, "Importance": importancia_var})
            st.session_state.df_importance_dict = df_imp.sort_values(by="Importance", ascending=False).to_dict(orient="records")
            st.dataframe(df_imp.sort_values(by="Importance", ascending=False))

           

    if st.session_state.df_importance_dict is not None:
        st.header("Analyzing the dataset with LLM:")
    
        if st.button("Analyze the results with LLM"):
            template=ChatPromptTemplate.from_messages([
                    ("system","You are a data science expert and provides your opinion about the results of the dataset. The dataset is the results of the random forest classifier, which shows the importance of each variable in the prediction of the model."),
                    ("user","Analyze the following dataset and share your opinion,and what it's about. Here is the dataset: {dataset}")
            ])
            chain = template | llm
            
            with st.spinner("LLM is thinking..."):
                response = chain.invoke({"dataset": st.session_state.df_importance_dict})
                st.markdown(response.content)
    
if __name__=="__main__":
    main()         