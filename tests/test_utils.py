import unittest
import os
import tempfile
from docx import Document
import PyPDF2
from utils import file_reader


class TestFileReader(unittest.TestCase):

    def test_txt_file_reading(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp:
            temp.write(b'Test content for txt file')
            temp.seek(0)
            contents = file_reader(temp)
            self.assertEqual(contents, b'Test content for txt file')

    def test_docx_file_reading(self):
        doc = Document()
        doc.add_paragraph('Test content for docx file')
        temp_path = tempfile.mktemp(suffix='.docx')
        doc.save(temp_path)
        with open(temp_path, 'rb') as temp:
            self.assertEqual(file_reader(temp), 'Test content for docx file')
        os.remove(temp_path)

    # def test_pdf_file_reading(self):
    #     pdf_writer = PyPDF2.PdfFileWriter()
    #     pdf_writer.write(pdf.PageObject.createBlankPage(None, 100, 100))
    #     # pdf_page = PyPDF2.pdf.PageObject.createBlankPage(None, 100, 100)
    #     temp_path = tempfile.mktemp(suffix='.pdf')
    #     with open(temp_path, 'wb') as output_pdf_file:
    #         pdf_writer.write(output_pdf_file)
    #     with open(temp_path, 'rb') as temp:
    #         self.assertEqual(file_reader(temp).strip(), '')
    #     os.remove(temp_path)

    # This test might be platform-dependent due to the usage of pywin32 and will only run on Windows
    @unittest.skipUnless(os.name == 'nt', "Skip .doc test for non-Windows")
    def test_doc_file_reading(self):
        # This test requires a more complex setup for the .doc file creation,
        # so you might want to either skip this test or manually create a
        # sample .doc file and test against it.

        pass


if __name__ == '__main__':
    unittest.main()
