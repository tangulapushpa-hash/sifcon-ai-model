FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

<<<<<<< HEAD
RUN pip install --upgrade pip

RUN pip install -r requirements.txt
=======
RUN pip install --no-cache-dir -r requirements.txt
>>>>>>> 869f2b8e9fba2329a9f8a2b8b5b0a6cf8e22bc89

COPY . .

EXPOSE 8501

<<<<<<< HEAD
CMD streamlit run app.py --server.port 8501 --server.address 0.0.0.0
=======
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0
>>>>>>> 869f2b8e9fba2329a9f8a2b8b5b0a6cf8e22bc89
