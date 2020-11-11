"""
Module for loading and handling DOW Jones data.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

from data_handling.global_data_params import DOW_JONES_RELATIVE_PATH, DEFAULT_LOCAL_PATH
from data_handling.data_utils import read_csv


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


def from_dow_jones_entries_to_numpy(dj_entries: DowJonesEntries) -> np.ndarray:
    """
    Converts the data from a DoWJonesEntries dataclass object to a numpy array of shape (N, D) where N is the number pf
    samples and =13 is the data dimension.
    :param dj_entries: a DowJonesEntries object containing data
    :return: numpy array of shape (N, 13)
    """
    arrays = [dj_entries.quarter, dj_entries.open, dj_entries.high, dj_entries.low, dj_entries.close, dj_entries.volume,
              dj_entries.percent_change_price, dj_entries.percent_change_next_weeks_price,
              dj_entries.percent_change_volume_over_last_wk, dj_entries.percent_return_next_dividend,
              dj_entries.next_weeks_open, dj_entries.next_weeks_close, dj_entries.previous_weeks_volume]
    return np.transpose(np.array(arrays))


if __name__ == "__main__":
    outdict = read_csv(DEFAULT_LOCAL_PATH / DOW_JONES_RELATIVE_PATH)
    djones_object = from_dict_to_dow_jones_entries(outdict)
    print(djones_object)
    new_array = from_dow_jones_entries_to_numpy(djones_object)
    print(np.shape(new_array))
