# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9

ADD main.py .

RUN pip install requests beautifulsoup4

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "main.py"]
