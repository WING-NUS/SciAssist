
from typing import List, Tuple

from bert_parscit import predict_for_string, predict_for_text, predict_for_pdf
import gradio as gr

def pred_str(input) -> List[Tuple[str, str]]:
    _, tokens_list, tags_list = predict_for_string(input)
    output = []
    for tokens,tags in zip(tokens_list, tags_list):
        for token,tag in zip(tokens, tags):
            output.append((token,tag))
        output.append(("\n", None))
    return output

def pred_file(input, dehyphen = False) -> List[Tuple[str, str]]:
    filename = input.name
    if filename[-4:] == ".txt":
        _, tokens_list, tags_list = predict_for_text(filename, dehyphen=dehyphen)
    elif filename[-4:] == ".pdf":
        _, tokens_list, tags_list = predict_for_pdf(filename, dehyphen=dehyphen)
    else:
        return [("File Format Error !", None)]
    output = []
    for tokens,tags in zip(tokens_list, tags_list):
        for token,tag in zip(tokens, tags):
            output.append((token,tag))
        output.append(("\n\n", None))
    return output


parse_str = gr.Interface(
    fn=pred_str,
    inputs=[
        gr.Textbox(
            label="String",
            lines=5,
        ),
        gr.Checkbox(label="de-hyphen")
    ],
    outputs=gr.HighlightedText(
        label="The Result of Parsing",
        combine_adjacent=True,
        adjacent_separator=" ",
    ),
    live=True,
    examples=[["Waleed Ammar, Matthew E. Peters, Chandra Bhagavat- ula, and Russell Power. 2017. The ai2 system at semeval-2017 task 10 (scienceie): semi-supervised end-to-end entity and relation extraction. In ACL workshop (SemEval).",True],
              ["Isabelle Augenstein, Mrinal Das, Sebastian Riedel, Lakshmi Vikraman, and Andrew D. McCallum. 2017. Semeval 2017 task 10 (scienceie): Extracting keyphrases and relations from scientific publications. In ACL workshop (SemEval).",False]]
)

parse_file = gr.Interface(
    fn=pred_file,
    inputs=[
        "file",
        gr.Checkbox(label="de-hyphen"),
    ],
    outputs=gr.HighlightedText(
        elem_id="htext",
        label="The Result of Parsing",
        combine_adjacent=True,
        adjacent_separator=" ",
    ),
    css="#htext span {white-space: pre-line}",
    live=True
)
demo = gr.TabbedInterface([parse_str, parse_file], ["Parsing reference string", "Parsing reference for PDF"])

parse_file.launch(share=True)