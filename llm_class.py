from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from openai import ChatCompletion, Embedding
import dotenv

dotenv.load_dotenv()
import os
import openai

openai.api_key = os.environ.get('OPENAI_API_KEY')


class LLMUtils:
    def __init__(self, model: ChatCompletion, embedding_source: FAISS, model_names: str = 'gpt-3.5-turbo'):
        self.model = model
        self.embedding_source = embedding_source
        self.similar_resumes = None  # placeholder until find_similar resumes runs
        self.model_names = model_names

    def find_similar_resumes(self, resume: str):
        """
        Use the FAISS vector db to find the resumes most similar and return the metadata of those 3 mot similar
        """
        vector = openai.Embedding.create(
            model="text-embedding-ada-002",
            input="The food was delicious and the waiter..."
        )
        vector = vector['data'][0]['embedding']
        # Assuming FAISS index returns ids and distances
        print(resume)
        print(len(vector))
        _ = self.embedding_source.search(resume, k=3, search_type='similarity')
        _ = [i.page_content for i in _]

        # This is a placeholder since we don't have a way to retrieve actual resumes from their ids.
        # You should replace this with appropriate retrieval.
        self.similar_resumes = _

    def format_resume_examples(self, section: str) -> str:
        """
        Using only the 3 most similar resumes, format the resumes so that they create a string that looks like this:
        """
        examples = []
        for idx, resume in enumerate(self.similar_resumes):
            # Again, this is a placeholder, you'd need to extract the actual section from the resume.
            section_example = "Extracted section here"

            example = f"""
Example {idx + 1}:

```resume
{resume}
```
```{section}
{section_example}
```
"""
            examples.append(example)
        return '\n'.join(examples)

    def extract_section(self, resume: str, section: str):
        """
        1. find similar examples for the section with FAISS
        2. prep the prompt with placeholders for examples
        3. format the examples into a string
        4. append the examples into the same string
        """
        self.find_similar_resumes(resume)
        self.similar_resumes = self.format_resume_examples(section)
        system_prompt = f"""
You are a resume parser, your job is to extract information from the resume and fill a pre-determined section based on that information. 
This time, that predetermined section is the {section} section.

Here are some examples of similar resumes being parsed for the {section} section.
{self.similar_resumes}
"""
        human_prompt = f"""
Here is the resume:
```resume
{resume}
```

You are writing for the {section} section. Do not return any thing other than the extracted section.
Always use human readable language. Do not use any special characters or formatting beyond bullets and paragraphs.
NEVER ADD DISCLAIMERS OR DISCLAIMERS TO THE EXTRACTED SECTION.
"""
        print(self.model)
        response = self.model.create(
            model=self.model_names,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt},
                {
                    "role": "user",
                    "content": human_prompt}
            ]
        )

        return response['choices'][0]['message']['content']  # return the text not the completion object
