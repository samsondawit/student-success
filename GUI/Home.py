import streamlit as st
from PIL import Image
from streamlit_extras.switch_page_button import switch_page
import os
# logo_path = Image.open('/mount/src/student-success/GUI/logo.png')
current_script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = Image.open(os.path.join(current_script_dir, 'logo.png'))
st.set_page_config(page_title="Home", page_icon=logo_path, layout="wide")

with st.sidebar:
        st.header("Select the language you want to read the home page in:")
        col1, col2 = st.columns([1, 4])
        if col1.button(label='EN'):
            st.session_state['language'] = 'EN'
        if col2.button(label='RU'):
            st.session_state['language'] = 'RU'
            
language = st.session_state.get('language', 'EN')

if language == 'RU':
    st.title('Предсказатель Академического Успеха')
    st.markdown("""
### Добро пожаловать в Academic Success Predictor - `основной шаг к персонализированному образованию`

Этот интерактивный инструмент предназначен для помощи преподавателям, студентам и администраторам в прогнозировании академических результатов на основе различных факторов. Используя передовые алгоритмы машинного обучения, это приложение может прогнозировать вероятность успеха студента, предоставляя ценные данные, которые могут информировать образовательные стратегии и вмешательства.

Это приложение является доказательством концепции того, что может быть возможным, если набор данных собран тщательно, что позволяет персонализировать и, в последствии, улучшить текущую модель образования, которую мы имеем.


### Вызов Проекта Цифровой Фараби
Этот проект разработан как часть [Вызова Проекта Цифровой Фараби](https://farabi.university/news/85336?lang=ru), Этап 2.
###### Разработчик: **Самсон Дауит Бекеле** | [LinkedIn](https://linkedin.com/in/samsondawit)
###### Руководитель: **Доцент, Иманкулов Тимур (PhD)**

Этот проект занял первое место на соревновании. Пресс-релиз университета можно найти [здесь](https://farabi.university/news/86373?lang=ru).
""", unsafe_allow_html=True)
    concept_presentation_path = os.path.join(current_script_dir, 'Concept presentation.pptx')
    UI_presentation_path = os.path.join(current_script_dir, 'UI presentation.pptx')
    st.markdown(""" #### Презентации: 
            """)

    col1, col2 = st.columns([2,7])
    with col1:
        with open(concept_presentation_path, "rb") as file:
            st.download_button(
                label="Презентация концепции",
                data=file,
                file_name="Concept presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
    with col2:
        with open(UI_presentation_path, "rb") as file:
            st.download_button(
                label="Презентация пользовательского интерфейса",
                data=file,
                file_name="UI Presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
    st.markdown("""
### Как это работает
- **Выберите модель**: Выберите из 4 предварительно обученных моделей, каждая из которых адаптирована для интерпретации различных наборов данных и улавливания различных индикаторов успеваемости студентов.
- **Ввод данных**: Введите данные студента из существующего набора данных, соответствующего модели, или введите вручную для прогнозирования.
- **Получите прогнозы**: Приложение обработает данные через выбранную прогностическую модель и предоставит вам вероятность академического успеха или неудачи.
- **Изучите визуализации**: После прогнозирования посетите вкладку Визуализация в боковой панели, чтобы увидеть соответствующие графики и рисунки, которые объясняют процесс принятия решений модели и дают информацию о наборе данных, на котором была обучена модель.

### Почему предсказывать академический успех?
Задачей было предложить курсы на основе предпочтений в обучении и академической успеваемости. Это широкая и активно исследуемая область, которая ставит значительные вызовы. Такая модель в настоящее время невозможна из-за отсутствия подходящего набора данных.
Поэтому мы предлагаем предсказывать академический успех, поскольку это `ключевой первый шаг к персонализированному образованию`. Понимая потенциальные исходы, образовательные работники могут:
- **Направлять вмешательства**: Выделить ресурсы и поддержку студентам, которым предсказана необходимость в дополнительной помощи.
- **Персонализировать обучение**: Адаптировать образовательный опыт к индивидуальным потребностям студента, максимизируя его шансы на успех.
- **Отслеживать прогресс**: Следить за вероятностью успеха со временем и корректировать стратегии по мере развития студентов и изменения обстоятельств.
- **Способствовать академическому росту**: Использовать полученные знания для создания среды, где каждый студент имеет возможность добиться успеха.

### Начало работы
1. Выберите прогностическую модель из 4 предварительно обученных моделей. Более подробная информация о моделях дана на странице прогнозов.
2. Выберите способ ввода данных в модель. Вы можете вручную ввести информацию о студенте или выбрать из предварительно загруженного набора данных.
3. После ввода данных нажмите кнопку "Предсказать" и наблюдайте, как модель оценивает вероятность академического успеха. Всё так просто!



""", unsafe_allow_html=True)
    st.sidebar.success("Выберите страницу прогнозов, чтобы продолжить.")
    if st.button("Перейти к прогнозам", key="predict_ru"):
            switch_page("Predictions")

else:
    st.title('Academic Success Predictor')
    st.markdown("""
                
### Welcome to the Academic Success Predictor - `a foundational step towards personalized education`

This interactive tool is designed to help educators, students, and administrators predict academic outcomes based on a variety of factors. Utilizing advanced machine learning algorithms, this app can forecast the likelihood of student success, providing valuable insights that can inform educational strategies and interventions.

This app is a proof of concept of what could be possible if a dataset is meticulously collected, enabling personalization and subsequently the improvement of the current education model we have. 


### Digital Farabi Project Challenge
This project is developed as part of the [Digital Farabi Project Challenge](https://farabi.university/news/85336?lang=en), Round 2. This challenge was sponsored by ASUS Education and Bugin Holding. 
###### Developer: **Samson Dawit Bekele** | [LinkedIn](https://linkedin.com/in/samsondawit)
###### Advisor: **Assoc. Prof., Imankulov Timur (PhD)**

This project has won the first place in the competition. The university press release can be found [here](https://farabi.university/news/86373?lang=en). 
""", unsafe_allow_html=True)
    concept_presentation_path = os.path.join(current_script_dir, 'Concept presentation.pptx')
    UI_presentation_path = os.path.join(current_script_dir, 'UI presentation.pptx')
    st.markdown(""" #### Presentations: 
            """)

    col1, col2 = st.columns([1,4])
    with col1:
        with open(concept_presentation_path, "rb") as file:
            st.download_button(
                label="Concept Presentation",
                data=file,
                file_name="Concept presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
    with col2:
        with open(UI_presentation_path, "rb") as file:
            st.download_button(
                label="UI Presentation",
                data=file,
                file_name="UI Presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
    st.markdown("""

    ### How It Works
    - **Select a Model**: Choose from 4 pre-trained models, each tailored to interpret different data sets and capture various indicators of student performance.
    - **Input Data**: Input student data from an existing dataset corresponding to the model or input manually to make predictions.
    - **Receive Predictions**: The app will process the data through the selected predictive model and provide you with the probability of academic success or failure.
    - **Explore Visualizations**: After making predictions, visit the Visualization tab in the sidebar to see relevant graphs and figures that explain the model's decision making and give information about the dataset the model was trained on. 

    ### Why Predict Academic Success?
    The given task was to recommend courses based on learning preference and academic performance. This is a wide and active research area that posits significant challenges. Such a model is not possible currently due to the lack of an appropriate dataset.
    Therefore, we propose predicting academic success because it is a `crucial first step towards personalized education`. By understanding potential outcomes, educators can:
    - **Target Interventions**: Allocate resources and support to students who are predicted to require additional assistance.
    - **Personalize Learning**: Tailor educational experiences to individual student needs, maximizing their chances for success.
    - **Monitor Progress**: Keep track of the likelihood of success over time and adjust strategies as students develop and circumstances change.
    - **Foster Academic Growth**: Use insights to create an environment where all students have the opportunity to succeed.

    ### Get Started
    1. Select a predictive model from the 4 pre-trained models. More detail about the models is given in the predictions page.
    2. Choose your method for inputting data into the model. You can either manually enter student information or select from a preloaded dataset.
    3. Once you've input the data, hit the "Predict" button and watch the model evaluate the likelihood of academic success. It's that simple!

    """, unsafe_allow_html=True)
        
    if st.button("Go to predictions", key="predict_en"):
            switch_page("Predictions")
        
                
    st.sidebar.success("Select the Predictions Page to proceed.")
