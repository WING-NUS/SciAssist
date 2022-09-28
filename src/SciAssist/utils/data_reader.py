import csv

from datasets import Dataset


def csv_reader(filename: str, encoding = 'ISO-8859-1'):
    """

    Args:
        filename (`str`): The name of the input CSV file

    Returns:
        data_set (`Dataset`):
            A dataset, for example:
                Dataset({
                    features: ['', 'paper_name', 'text', 'summary', 'paper_id',...],
                    num_rows: 4611
                })

    """
    data_set = {}
    with open(filename, 'r', newline='', encoding=encoding) as f:
        rows = csv.reader(f)
        # Get Column names
        keys = next(rows)
        for key in keys:
            data_set[key] = []
        # Add values by column
        for row in rows:
            for id, key in enumerate(keys):
                data_set[key].append(row[id])

    # Remove columns that aren't involved in the input of model
    # in case of ValueError: Unable to create tensor,
    # you should probably activate truncation and/or
    # padding with 'padding=True' 'truncation=True'
    # to have batched tensors with the same length.
    if "" in data_set.keys():
        data_set.pop("")
    data_set = Dataset.from_dict(data_set)
    return data_set
