from typing import List, Tuple

import gradio as gr

from src.pipelines.bert_parscit import predict_for_text, predict_for_pdf, predict_for_string

description = '''
If you'd like to generate a demo like this on your own, please go for [**our GitHub repo**](https://github.com/ljhgabe/BERT-ParsCit) 
and try the following codes.

This is the command we actually run:
```python

from typing import List, Tuple
from bert_parscit import  predict_for_text, predict_for_pdf
import gradio as gr

def pred_file(input, dehyphen = False) -> List[Tuple[str, str]]:

    if input == None:
        return None
    filename = input.name
    #Identify the format of input and parse reference strings
    if filename[-4:] == ".txt":
        _, tokens_list, tags_list = predict_for_text(filename, dehyphen=dehyphen)
    elif filename[-4:] == ".pdf":
        _, tokens_list, tags_list = predict_for_pdf(filename, dehyphen=dehyphen)
    else:
        return [("File Format Error !", None)]
    #Prepare for the input gradio.HighlightedText accepts.
    output = []
    for tokens,tags in zip(tokens_list, tags_list):
        for token,tag in zip(tokens, tags):
            output.append((token,tag))
        output.append(("\n\n", None))
    return output

with gr.Blocks(css="#htext span {white-space: pre-line}") as demo:
    gr.Markdown("# Gradio Demo for BERT-ParsCit")
    gr.Markdown("Gradio Demo for BERT-ParsCit. To use it, simply upload your .txt or .pdf file.")
    with gr.Row():
        with gr.Column():
            file = gr.File()
            dehyphen = gr.Checkbox(label="dehyphen")
            with gr.Row():
                btn = gr.Button("Parse")

        output = gr.HighlightedText(
            elem_id="htext",
            label="The Result of Parsing",
            combine_adjacent=True,
            adjacent_separator=" ",
        )
    gr.Markdown(value=description)
    btn.click(
        fn=pred_file,
        inputs=[file, dehyphen],
        outputs=output
    )

demo.launch(share=True)
```
'''


def pred_str(input, dehyphen=False) -> List[Tuple[str, str]]:
    _, tokens_list, tags_list = predict_for_string(input, dehyphen=dehyphen)
    output = []
    for tokens, tags in zip(tokens_list, tags_list):
        for token, tag in zip(tokens, tags):
            output.append((token, tag))
        output.append(("\n\n", None))
    return output


def pred_file(input, dehyphen=False) -> List[Tuple[str, str]]:
    if input == None:
        return None
    filename = input.name
    # Identify the format of input and parse reference strings
    if filename[-4:] == ".txt":
        _, tokens_list, tags_list = predict_for_text(filename, dehyphen=dehyphen)
    elif filename[-4:] == ".pdf":
        _, tokens_list, tags_list = predict_for_pdf(filename, dehyphen=dehyphen)
    else:
        return [("File Format Error !", None)]
    # Prepare for the input gradio.HighlightedText accepts.
    output = []
    for tokens, tags in zip(tokens_list, tags_list):
        for token, tag in zip(tokens, tags):
            output.append((token, tag))
        output.append(("\n\n", None))
    return output


with gr.Blocks(css="#htext span {white-space: pre-line}") as demo:
    gr.Markdown("# Gradio Demo for BERT-ParsCit")
    with gr.Tabs():
        with gr.TabItem("Parse Reference for File"):
            gr.Markdown("To use it, simply upload your .txt or .pdf file.")
            with gr.Row():
                with gr.Column():
                    file = gr.File()
                    file_dehyphen = gr.Checkbox(label="dehyphen")
                    with gr.Row():
                        file_btn = gr.Button("Parse")

                file_output = gr.HighlightedText(
                    elem_id="htext",
                    label="The Result of Parsing",
                    combine_adjacent=True,
                    adjacent_separator=" ",
                )
        with gr.TabItem("Parse Reference for String"):
            gr.Markdown("To use it, simply input one or more strings.")
            with gr.Row():
                with gr.Column():
                    str = gr.Textbox(label="Input String")
                    str_dehyphen = gr.Checkbox(label="dehyphen")
                    with gr.Row():
                        str_btn = gr.Button("Parse")
                str_output = gr.HighlightedText(
                    elem_id="htext",
                    label="The Result of Parsing",
                    combine_adjacent=True,
                    adjacent_separator=" ",
                )
            str_examples = gr.Examples(examples=[[
                                                     "Waleed Ammar, Matthew E. Peters, Chandra Bhagavat- ula, and Russell Power. 2017. The ai2 system at semeval-2017 task 10 (scienceie): semi-supervised end-to-end entity and relation extraction. In ACL workshop (SemEval).",
                                                     True],
                                                 [
                                                     "Isabelle Augenstein, Mrinal Das, Sebastian Riedel, Lakshmi Vikraman, and Andrew D. McCallum. 2017. Semeval 2017 task 10 (scienceie): Extracting keyphrases and relations from scientific publications. In ACL workshop (SemEval).",
                                                     False]], inputs=[str, str_dehyphen])

        with gr.TabItem("Source Code"):
            gr.Markdown(value=description)

    file_btn.click(
        fn=pred_file,
        inputs=[file, file_dehyphen],
        outputs=file_output
    )
    str_btn.click(
        fn=pred_str,
        inputs=[str, str_dehyphen],
        outputs=str_output
    )

demo.launch(share=True)