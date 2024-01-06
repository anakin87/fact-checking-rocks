FROM deepset/haystack:base-cpu-v1.23.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# copy only the application files in /app
# Streamlit does not allow running an app from the root directory
COPY Rock_fact_checker.py app/
COPY pages app/pages
COPY app_utils app/app_utils
COPY data app/data

WORKDIR app

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Rock_fact_checker.py"]
