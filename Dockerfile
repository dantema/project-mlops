FROM python3.9

COPY . 
CMD pip install -r requirements.txt

CMD ["python3", "-m", "mainflow.py"]