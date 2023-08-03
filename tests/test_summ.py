import json

import pytest

from SciAssist import Summarization


# Summ
@pytest.fixture
def BERTSummGrobid():
    res = []
    with open("output/grobid/BERT_paper_summ_bart.json", "r", encoding="utf-8") as f:
        results = f.readlines()
        for i in results:
            r = i.replace("\n","")
            res.append(json.loads(r))
    return res[0]

@pytest.fixture
def BERTSummPDFMiner():
    res = []
    with open("output/pdfminer-six/BERT_paper_summ_bart.json", "r", encoding="utf-8") as f:
        results = f.readlines()
        for i in results:
            r = i.replace("\n", "")
            res.append(json.loads(r))
    return res[0]


class TestSumm():

    def test_SummGrobid(self, BERTSummGrobid):
        parser = Summarization()

        res = parser.predict("pdfs/BERT_paper.pdf")
        assert res["summary"] == BERTSummGrobid["summary"]


    def test_SummPDFMiner(self, BERTSummPDFMiner):
        parser = Summarization(os_name="nt")
        res = parser.predict("pdfs/BERT_paper.pdf")
        assert res["summary"] == BERTSummPDFMiner["summary"]




