import os

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from SciAssist import BASE_OUTPUT_DIR

path="test.pdf"
def process_pdf(path):
    raw_text = []
    res = {
        "title":"",
        "author":[],
        "body_text":[],
        "reference":[]
    }
    for page_layout in extract_pages(path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().replace("\n","")
                text = text.strip()
                if text.isdigit() == False and text != "":
                    raw_text.append(text)

    res["title"] = raw_text[0]

    i = 1 # the index of the current text in raw_text

    while i<len(raw_text):
        if raw_text[i].lower()=="abstract":
            i += 1
            break
        res["author"].append(raw_text[i])
        i += 1

    while i<len(raw_text):
        if "Introduction" in raw_text[i]:
            break
        i += 1

    while i<len(raw_text):
        if raw_text[i].lower() in ["references","reference"]:
            i += 1
            break
        res["body_text"].append(raw_text[i])
        i += 1

    while i<len(raw_text):
        if "Appendix" in raw_text[i]:
            i += 1
            break
        res["reference"].append(raw_text[i])
        i += 1

    return res


def windows_get_reference(path, output_dir: str = BASE_OUTPUT_DIR):

    os.makedirs(output_dir, exist_ok=True)

    res = process_pdf(path)
    output_path = os.path.join(output_dir, os.path.basename(path)[:-4] + "_ref.txt")
    with open(output_path,"w",encoding="utf-8") as output:
        for i in res["reference"]:
            output.write(i+"\n")
    return output_path

def windows_get_bodytext(path, output_dir: str = BASE_OUTPUT_DIR):

    os.makedirs(output_dir, exist_ok=True)

    res = process_pdf(path)
    output_path = os.path.join(output_dir, os.path.basename(path)[:-4] + "_body.txt")
    with open(output_path,"w",encoding="utf-8") as output:
        for i in res["body_text"]:
            output.write(i+" ")
    return output_path
