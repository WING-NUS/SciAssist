
from typing import List, Tuple
from src.pipelines.bert_parscit import  predict_for_text, predict_for_pdf
import gradio as gr
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