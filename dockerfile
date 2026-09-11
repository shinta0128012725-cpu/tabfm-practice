FROM python:3.12

WORKDIR /app

COPY tabfm/ ./tabfm/
WORKDIR /app/tabfm
RUN pip install -e .[pytorch,examples]

CMD ["bash"]