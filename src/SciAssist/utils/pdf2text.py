import argparse
import json
import os
from typing import List

import nltk

from SciAssist import BASE_OUTPUT_DIR, BASE_TEMP_DIR
from SciAssist.bin.doc2json.doc2json.grobid2json.process_pdf import process_pdf_file


def get_reference(json_file: str, output_dir: str = BASE_OUTPUT_DIR):

    os.makedirs(output_dir, exist_ok=True)
    assert json_file[-4:] == "json"
    output_path = os.path.join(output_dir,os.path.basename(json_file)[:-5]+"_ref.txt")

    strings_file = open(output_path,"w")

    with open(json_file,'r') as f:
        data = json.load(f)
        for _, context in data["pdf_parse"]["bib_entries"].items():
            raw_text = context["raw_text"]
            if raw_text != "":
                strings_file.write(raw_text)
                strings_file.write("\n")

    strings_file.close()
    return output_path


def get_bodytext(json_file: str, output_dir: str = BASE_OUTPUT_DIR, suffix="_body"):

    os.makedirs(output_dir, exist_ok=True)
    assert json_file[-4:] == "json"
    output_path = os.path.join(output_dir,os.path.basename(json_file)[:-5]+ suffix + ".txt")

    strings_file = open(output_path,"w")

    with open(json_file,'r') as f:
        data = json.load(f)
        for context in data["pdf_parse"]["body_text"]:
            sent = context["text"]
            if sent != "":
                sent = nltk.word_tokenize(sent)
                strings_file.write(" ".join(sent))
                strings_file.write(" ")

    strings_file.close()
    return output_path

def get_text_by_section(json_file: str, section: List, output_dir: str = BASE_OUTPUT_DIR, suffix=None):
    os.makedirs(output_dir, exist_ok=True)
    assert json_file[-4:] == "json"
    if suffix is None:
        suffix = "_" + "_".join(section)
    output_path = os.path.join(output_dir, os.path.basename(json_file)[:-5] + suffix + ".txt")
    strings_file = open(output_path,"w")

    section = [s.lower() for s in section]
    with open(json_file,'r') as f:
        data = json.load(f)
        for context in data["pdf_parse"]["body_text"]:
            sent = context["text"]
            for j in section:
                if sent != "" and j in context["section"].lower():
                    sent = nltk.word_tokenize(sent)
                    strings_file.write(" ".join(sent))
                    strings_file.write(" ")
                else:
                    continue

    strings_file.close()
    return output_path

def get_IC(json_file: str, output_dir: str = BASE_OUTPUT_DIR, suffix=None):
    os.makedirs(output_dir, exist_ok=True)
    section=["Introduction","Conclusion"]
    assert json_file[-4:] == "json"
    if suffix is None:
        suffix = "_" + "_".join(section)
    output_path = os.path.join(output_dir, os.path.basename(json_file)[:-5] + suffix + ".txt")
    strings_file = open(output_path,"w")
    res=[]
    section = [s.lower() for s in section]
    flag=0
    with open(json_file,'r') as f:
        data = json.load(f)
        for context in data["pdf_parse"]["body_text"]:
            sent = context["text"]
            for j in section:
                if sent != "" and j in context["section"].lower():
                    if j=="introduction":
                        flag=1
                    sent = nltk.word_tokenize(sent)
                    res.extend(sent)
                else:
                    continue
        if flag==0:
            res=[]
            for context in data["pdf_parse"]["body_text"]:
                sent = context["text"]
                if sent != "":
                    sent = nltk.word_tokenize(sent)
                    res.extend(sent)
    strings_file.write(" ".join(res[:800]))
    strings_file.write(" ")
    strings_file.close()
    return output_path

def get_abs(json_file: str, output_dir: str = BASE_OUTPUT_DIR, suffix="_abs"):

    os.makedirs(output_dir, exist_ok=True)
    assert json_file[-4:] == "json"
    output_path = os.path.join(output_dir,os.path.basename(json_file)[:-5]+ suffix + ".txt")

    strings_file = open(output_path,"w")

    with open(json_file,'r') as f:
        data = json.load(f)
        for context in data["pdf_parse"]["abstract"]:
            sent = context["text"]
            if sent != "":
                sent = nltk.word_tokenize(sent)
                strings_file.write(" ".join(sent))
                strings_file.write(" ")

    strings_file.close()
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracting strings from JSON")
    parser.add_argument("--input_file", required=True, help="path to the input json file")
    parser.add_argument("--temp_dir", default=BASE_TEMP_DIR, help="path to the temp dir for putting json files")
    parser.add_argument("--output_dir", default=BASE_OUTPUT_DIR, help="path to the output dir for putting text files")
    parser.add_argument("--save_temp", action="store_true", default=True, help="to save temporary json files")

    parser.add_argument("--reference", action="store_true", default=False, help="to extract reference strings")
    parser.add_argument("--bodytext", action="store_true", default=False, help="to extract bodytext")

    args = parser.parse_args()
    pdf_file = args.input_file
    output_dir = args.output_dir
    temp_dir = args.temp_dir


    json_file = process_pdf_file(input_file=pdf_file, temp_dir=temp_dir, output_dir=temp_dir)

    if args.reference:
        ref = get_reference(json_file=json_file, output_dir=output_dir)

    if args.bodytext:
        ref = get_bodytext(json_file=json_file, output_dir=output_dir)