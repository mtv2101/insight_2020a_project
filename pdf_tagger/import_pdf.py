
import PyPDF2
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import time


def get_pdf_text(path, library='pdfminer'):

    if library == 'pypdf2':

        start_time = time.time()

        pdfFileObj = open(path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        print(pdfReader.numPages)
        pageObj = pdfReader.getPage(0)
        text = pageObj.extractText()

        print('pdf-text conversion took ' + str(time.time() - start_time) + ' seconds')

        return text

    elif library == 'pdfminer':

        start_time = time.time()

        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8' # formerly used in TextConverter
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        fp = open(path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        page_text = []

        for page in PDFPage.get_pages(fp,
                                      pagenos,
                                      maxpages=maxpages,
                                      password=password,
                                      caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)

            page_text.append(retstr.getvalue())

        fp.close()
        device.close()
        retstr.close()

        print('pdf-text conversion took ' + str(time.time() - start_time) + ' seconds')

        return page_text

    else:
        print('improperly specified pdf conversion library - check spelling')



