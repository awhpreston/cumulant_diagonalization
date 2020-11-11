"""
Common functions for data handling.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List


def read_csv(file_path: Path) -> Dict[str, List[Any]]:
    """
    Reads a comma-separated-variables file to a dictionary whose keys are the first row entries and whose values
    are lists of row entries for each column.
    :param file_path: path to the csv file
    :return: {first row entry: [subsequent row entries]}
    """
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        row_num = 0
        output_dict = dict()
        key_map = dict()  # maps column index to dict key
        for row in reader:
            entries = [entry for entry in row]
            if row_num == 0:
                for i, entry in enumerate(entries):
                    key_map[i] = entry
                    output_dict[entry] = list()
            else:
                for i, entry in enumerate(entries):
                    key = key_map[i]
                    output_dict[key].append(entry)
            row_num += 1
    return output_dict
