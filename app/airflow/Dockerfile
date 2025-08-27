FROM apache/airflow:2.10.1

# 1) Install OS deps as root (optional, but useful for psycopg2, numpy builds, etc.)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

# 2) Switch to airflow user BEFORE pip install
USER airflow

# Copy requirements from the repo root into the image, owned by airflow
# (compose build context is the repo root; adjust the path if your file lives elsewhere)
COPY --chown=airflow:0 requirements.txt /requirements.txt

# 3) Install Python packages as 'airflow' user
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install --no-cache-dir apache-airflow-providers-postgres
