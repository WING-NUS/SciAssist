TAG_NAMES = [
    "B-DATA",
    "I-DATA",
    "O",
]

num_tags = len(TAG_NAMES)

id2tag = {str(i): tag for i, tag in enumerate(TAG_NAMES)}
tag2id = {v: k for k, v in id2tag.items()}
