FROM deepset/haystack:base-cpu-v1.23.0

COPY requirements.txt .
RUN pip install -r requirements.txt

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# copy only the application files in /app
# Streamlit does not allow running an app from the root directory
COPY --chown=user Rock_fact_checker.py $HOME/app/
COPY --chown=user pages $HOME/app/pages
COPY --chown=user app_utils $HOME/app/app_utils
COPY --chown=user data $HOME/app/data

WORKDIR $HOME/app

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Rock_fact_checker.py"]
