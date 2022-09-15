#!/bin/bash
cd bin/doc2json
python setup.py develop
cd ../..
bash bin/doc2json/scripts/setup_grobid.sh
nohup bash bin/doc2json/scripts/run_grobid.sh >/dev/null 2>grobid_log &

