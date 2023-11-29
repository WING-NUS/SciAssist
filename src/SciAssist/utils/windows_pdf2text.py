import os

from pdfminer.high_level import extract_pages

from SciAssist import BASE_OUTPUT_DIR

path="test.pdf"
def process_pdf(path):
    raw_text = []
    res = {
        "title":"",
        "author":[],
        "abstract":[],
        "body_text":[],
        "reference":[]
    }
    print(path)
    fp = open(path, 'rb')
    for page_layout in extract_pages(path):
        for element in page_layout:
            if "get_text" in dir(element):
                text = element.get_text().replace("-\n","") # Removes hyphen from a word split between lines
                text = text.replace("\n"," ") # Replace "\n" with space (between words)
                text = text.strip()
                if text.isdigit() == False and text != "":
                    raw_text.append(text)
    fp.close()
    res["title"] = raw_text[0]

    i = 1 # the index of the current text in raw_text

    while i<len(raw_text):
        if raw_text[i].strip().replace(" ","")[:8] in ["abstract","Abstract", "ABSTRACT"]:
            i += 1
            break
        res["author"].append(raw_text[i])
        i += 1

    while i<len(raw_text):
        if raw_text[i].strip()[:2] in ["1 ", "1."] or \
                raw_text[i].strip().replace(" ","")[:12] in ["Introduction","INTRODUCTION"]:
            break
        res["abstract"].append(raw_text[i])
        i += 1

    while i<len(raw_text):
        if raw_text[i].strip().replace(" ","")[:9] in ["Reference", "REFERENCE"]:
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
