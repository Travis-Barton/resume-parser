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
        _ = self.embedding_source.search(resume, k=3, search_type='similarity')
        _ = [i.metadata for i in _]

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
            example = f"""
Example {idx + 1}:

```resume
{resume['resume']}
```
```{section}
{resume[section]}
```
"""
            examples.append(example)
        return '\n'.join(examples) + '\nThese are examples, never include them in the extracted section. Also never use tick marks in the extracted section.'

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

You are writing for the {section} section. Do not return any thing other than the extracted section.
Always use human readable language. Do not use any special characters or formatting beyond bullets and paragraphs.
NEVER ADD DISCLAIMERS OR DISCLAIMERS TO THE EXTRACTED SECTION.


{'NOTE: PROFESSIONAL EXPERIENCE does NOT include education and does NOT include a list of skills. Do not inlcude these in the extracted text.' if section == 'professional_experience' else ''}
{'NOTE: The label for each job title should be seperated from the date with enough tabs to put it at the end of the line. eg: "Neat Company, Remote                                                                                                                  8/2018-12/2019"' if section == 'professional_experience' else ''}
{'Note: Skills and Tech only includes bullet points of skills and tech and does not include work experience or project contributions' if section == 'skills_and_tech' else ''}
Here are some rules:
1. Do not return any thing other than the extracted section. That includes headers, footers, other sections, etc.
2. Always use • for bullets and paragraphs for paragraphs.
3. Never include disclaimers or disclaimers in the extracted section.
4. Only extract the section you are asked to extract. Do not extract other sections. (e.g. if you are asked to extract the Skills And Tech section, do not extract the Experience section) that will be taken care of by other parsers.
5. Always use • and never use - for bullets.

"""
        human_prompt = f"""
Here is the resume:
```resume
{resume}
```
"""
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
        print(system_prompt)
        return response['choices'][0]['message']['content']  # return the text not the completion object
