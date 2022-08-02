LABEL_NAMES = [
    'citation-number',
    'collection-title',
    'container-title',
    'doi',
    'issue',
    'number-of-pages',
    'volume',
    'issued',
    'year',
    'month',
    'day',
    'author',
    'editor',
    'edition',
    'page',
    'publisher',
    'title',
    'url',
    'year-suffix'
]

num_labels = len(LABEL_NAMES)

id2label = {str(i): label for i, label in enumerate(LABEL_NAMES)}
label2id = {v: k for k, v in id2label.items()}
