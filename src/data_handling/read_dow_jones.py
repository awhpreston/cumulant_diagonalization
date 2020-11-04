"""
Module for loading and handling DOW Jones data.
"""

import csv
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

DOW_JONES_RELATIVE_PATH = "dow_jones/dow_jones_index.data"


@dataclass
class DowJonesEntries:
    quarter: List[int]
    stock: List[str]
    date: List[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    percent_change_price: np.ndarray
    percent_change_volume_over_last_wk: np.ndarray
    previous_weeks_volume: np.ndarray
    next_weeks_open: np.ndarray
    next_weeks_close: np.ndarray
    percent_change_next_weeks_price: np.ndarray
    days_to_next_dividend: np.ndarray
    percent_return_next_dividend: np.ndarray


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


def from_dict_to_dow_jones_entries(data_dict: Dict[str, List[Any]]) -> DowJonesEntries:
    """
    From the output of read_csv, fill in a DowJonesEntries object.
    :param data_dict: DOW Jones data in a dictionary from read_csv
    :return: data sorted into a DowJonesEntries object
    """
    naive_dataclass = DowJonesEntries(**data_dict)
    percent_change_volume_over_last_wk = list()
    for percent in naive_dataclass.percent_change_volume_over_last_wk:
        if percent != '':
            percent_change_volume_over_last_wk.append(float(percent))
        else:
            percent_change_volume_over_last_wk.append(0.)
    previous_weeks_volume = list()
    for volume in naive_dataclass.previous_weeks_volume:
        if volume != '':
            previous_weeks_volume.append(int(volume))
        else:
            previous_weeks_volume.append(0)
    dj_object = DowJonesEntries(quarter=[int(quarter) for quarter in naive_dataclass.quarter],
                                stock=naive_dataclass.stock,
                                date=naive_dataclass.date,
                                open=np.array([float(price[1:]) for price in naive_dataclass.open]),
                                high=np.array([float(price[1:]) for price in naive_dataclass.high]),
                                low=np.array([float(price[1:]) for price in naive_dataclass.low]),
                                close=np.array([float(price[1:]) for price in naive_dataclass.close]),
                                volume=np.array([int(volume) for volume in naive_dataclass.volume]),
                                percent_change_price=np.array([float(percent) for percent in
                                                               naive_dataclass.percent_change_price]),
                                percent_change_next_weeks_price=np.array([float(percent) for percent in
                                                                          naive_dataclass.
                                                                         percent_change_next_weeks_price]),
                                percent_change_volume_over_last_wk=np.array(percent_change_volume_over_last_wk),
                                percent_return_next_dividend=np.array([float(percent) for percent in
                                                                       naive_dataclass.percent_return_next_dividend]),
                                days_to_next_dividend=np.array([int(days) for days in
                                                               naive_dataclass.days_to_next_dividend]),
                                next_weeks_open=np.array([float(price[1:]) for price in
                                                          naive_dataclass.next_weeks_open]),
                                next_weeks_close=np.array([float(price[1:]) for price in naive_dataclass.close]),
                                previous_weeks_volume=np.array(previous_weeks_volume))
    return dj_object


if __name__ == "__main__":
    outdict = read_csv(Path("local_data/"+DOW_JONES_RELATIVE_PATH))
    djones_object = from_dict_to_dow_jones_entries(outdict)
    print(djones_object)
