import os
import platform
from subprocess import run

PACKAGE_ROOT = os.path.dirname(__file__)
SCRIPTS_DIR = {'grobid': os.path.join(PACKAGE_ROOT, 'doc2json/scripts/')}

def setup_grobid():
    if platform.system().lower()=="linux":
        script = os.path.join(SCRIPTS_DIR['grobid'], 'setup_grobid_linux.sh')
        run(['bash',script])
    print("Grobid is installed.")


def run_grobid():
    if platform.system().lower()=="linux":
        script = os.path.join(SCRIPTS_DIR['grobid'], 'run_grobid_linux.sh')
        cmd = "nohup bash " + script + " >/dev/null 2>grobid_log &"
        run(cmd, shell=True)
    print("Grobid is running now.")

