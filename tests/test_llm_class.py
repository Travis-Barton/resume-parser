import unittest
import os
from mock import Mock  # This is a library to mock objects
from llm_class import LLMUtils  # Adjust the import according to where your class resides
from utils import file_reader
from langchain.docstore.document import Document

# Mock the FAISS and ChatOpenAI classes


class MockFAISS:
    def search(self, vector, k, search_type):
        # Returns mock ids and distances
        return [Document(page_content="Sample resume content"),
                Document(page_content="Sample resume content"),
                Document(page_content="Sample resume content")]


class MockChatOpenAI:
    def create(self, model, messages):
        # mock a return that allows for this: response['choices'][0]['message']['content']
        return {'choices': [{'message': {'content': 'Sample extracted section'}}]}


class LLMUtilsTests(unittest.TestCase):

    def setUp(self):
        self.model = MockChatOpenAI()
        self.embedding_source = MockFAISS()
        self.utils = LLMUtils(self.model, self.embedding_source)

    def test_find_similar_resumes(self):
        resume = "Sample resume content"
        self.utils.find_similar_resumes(resume)
        self.assertEqual(len(self.utils.similar_resumes), 3)

    def test_format_resume_examples(self):
        section = "Sample section"
        self.utils.similar_resumes = ["Sample resume content", "Sample resume content", "Sample resume content"]
        formatted = self.utils.format_resume_examples(section)
        self.assertTrue("Example 1:" in formatted)
        self.assertTrue("Example 2:" in formatted)
        self.assertTrue("Example 3:" in formatted)

    def test_extract_section(self):
        resume_path = os.path.join("..", "profiles", "Abbas Sheikh", "DiceResumeCVABBASSHEIKH+1.docx")
        with open(resume_path, 'rb') as resume_file:
            resume_content = file_reader(resume_file)

        section_content = self.utils.extract_section(resume_content, "Education")
        self.assertEqual(section_content, "Sample extracted section")


if __name__ == "__main__":
    unittest.main()
