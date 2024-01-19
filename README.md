# SmartDocInterpreter
Unlock the power of knowledge within your PDF documents with PDFInsightAI, the revolutionary app that utilizes cutting-edge artificial intelligence to provide unparalleled insights. Say goodbye to static PDFs and embrace a new era of dynamic understanding.

#### Setup:
- Clone the repo and create a virtual python env  (recommended) by using following command:-

``` bash
python3 -m venv venv
source venv/bin/activate
```

- After creating and activating your env run:-
```bash
pip install -r requirements.txt 
python -m spacy download en_core_web_md
```
#### Running auto_register
- With your active python env. Clone git repo ```https://github.com/Mayur-Thapliyal/SmartDocInterpreter/tree/dev``` and redirect to ITresumeParser and run
```python
    streamlit run SmartDocInterpreter.py
```

- You can find a sample PDF in sample_pdf as sample_pdf

This will run your code in your localhost (127.0.0.1:8501) by default you can change the port by giveing your port no after the command 




Thank You !

Contact mail : mayurthapliyal191@gmail.com