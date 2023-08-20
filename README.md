# AIM Resume Parser

This app translate resumes from DOCX and PDF into a specific DOC format.

## Getting Started

To use this you need to install the following dependencies:
```
pip install -r requirements.txt
```

You will also need an [API key from OpenAI](https://platform.openai.com/).

To run this code, simply run the following command:
```
streamlit run main.py
```

To recreate the vectorstore that powers the dynamic multi-shot prompting you will need to run the file `index_resumes.py`
```
python index_resumes.py
```

## Built With

* [Streamlit](https://streamlit.io/) - The web framework used
* [OpenAI](https://openai.com/) - The AI used
* [Travis Barton Consulting](https://travisbarton.com/) - The company that built this

All tests can be found in the `tests` folder.


## todo

1. [] Bold the work location
2. [] Add the applicant's name to the top of the resume
3. [] shrink the spacing at the title to 0
