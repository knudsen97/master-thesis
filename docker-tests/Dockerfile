FROM pytorch/pytorch:latest

WORKDIR /docker-tests

RUN apt-get update && apt-get install -y wget unzip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]