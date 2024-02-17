from SciAssist import ReferenceEntityLinkingLocal, ReferenceEntityLinkingOnline

def check_local():
    pipeline = ReferenceEntityLinkingLocal()
    files_dir = './output/sample_papers'
    pipeline.predict(files_dir)

def check_local2():
    pipeline = ReferenceEntityLinkingLocal()
    files_dir = './output/sample_papers'
    ref_lines = pipeline._batch_pdf_to_referencess(files_dir)
    save_file = './output/ref_data.txt'
    pipeline.utils.json_list_to_file(ref_lines, save_file)
    data = pipeline.utils.file_to_json_list(save_file)
    result = pipeline._ref_clustering(data)
    pipeline.utils.json_list_to_file(result, './output/local_result.txt')

def check_online():
    pipeline = ReferenceEntityLinkingOnline()
    files_dir = './output/sample_papers'
    pipeline.predict(files_dir)

def check_online2():
    pipeline = ReferenceEntityLinkingOnline()
    title_list = [
        'Automatic Generation of Citation Texts in Scholarly Papers: A Pilot Study',
        'Automatic related work section generation: experiments in scientific document abstracting',
        'BACO: A Background Knowledge- and Content-Based Framework for Citing Sentence Generation',
    ]
    pipeline.batch_title_to_references(title_list)

if __name__ == "__main__":
    check_local()
    check_online()