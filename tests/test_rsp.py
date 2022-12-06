import json

import pytest

from SciAssist import ReferenceStringParsing


# Rsp
@pytest.fixture
def BERTRspGrobid():
    res = []
    with open("output/grobid/BERT_paper_rsp_scibert.json", "r", encoding="utf-8") as f:
        results = f.readlines()
        for i in results:
            r = i.replace("\n","")
            res.append(json.loads(r))
    return res


@pytest.fixture
def BERTRspPDFMiner():
    res = []
    with open("output/pdfminer-six/BERT_paper_rsp_scibert.json", "r", encoding="utf-8") as f:
        results = f.readlines()
        for i in results:
            r = i.replace("\n","")
            res.append(json.loads(r))
    return res

class TestRsp():

    def test_RspGrobid(self, BERTRspGrobid):

        parser = ReferenceStringParsing()
        res = parser.predict("pdfs/BERT_paper.pdf")
        assert res == BERTRspGrobid


    def test_RspPDFMiner(self, BERTRspPDFMiner):
        parser = ReferenceStringParsing(os_name="nt")
        res = parser.predict("pdfs/BERT_paper.pdf")
        assert res == BERTRspPDFMiner


