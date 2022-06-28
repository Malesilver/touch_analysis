import re
import csv
import numpy as np
import os
from typing import List
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import argparse
from matplotlib import patches
from phase_utilities import *

file_dir = os.path.dirname(__file__)  # the directory that class "option" resides in
pd.set_option('display.max_columns', None)

MCT_DELTAGEN_DATA = 0
SCTY_DELTAGEN_DATA = 1
SCTX_DELTAGEN_DATA = 2

HEADER_ETS = [
    [MCT_DELTAGEN_DATA, "mct_deltas.values[0][0]", "mct_deltas", 0, 0],
    [SCTY_DELTAGEN_DATA, "sct_row_deltas[0]", "sct_row", 0, 0],
    [SCTX_DELTAGEN_DATA, "sct_col_deltas[0]", "sct_col", 0, 0],
]


class ETS_Dataframe:
    def __init__(self, file_path=None, Header_index=None):

        self.mct_grid = None
        self.sct_row = None
        self.sct_col = None
        self.row_num = None
        self.col_num = None
        self.Header_index = Header_index

        self.file_ext = os.path.basename(file_path).split(".")[-1]
        if self.file_ext == "csv":
            self.data_init = self.load_data_from_ets_csv(file_path)
        elif self.file_ext == "txt":
            pass
        elif self.file_ext == "json":
            pass

    def load_data_from_ets_csv(self, file_path):
        mutual_raw = []
        col_raw = []
        row_raw = []
        with open(file_path) as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
            for csv_line_idx, csv_line in enumerate(reader):
                # Split comma separated data in line into list
                # in case line end with space ""
                if csv_line[-1] == "":
                    csv_data = csv_line[:-1]
                else:
                    csv_data = csv_line

                # Process header at first line
                if csv_line_idx == 0:
                    header = csv_data
                    # Search header in first row to identify column indices for playback data
                    # print(csv_data[3])
                    for column_idx, column_str in enumerate(csv_line):
                        # print(column_str)
                        for idx, playback_data in enumerate(self.Header_index):
                            # print(playback_data[1])
                            if playback_data[1] in column_str:
                                # print("Found start of " + column_str + " idx=" + str(column_idx))
                                # print("Looking for end at " + str(playback_data[2]))
                                playback_data[3] = int(column_idx)

                            elif playback_data[2] in column_str:
                                self.Header_index[idx][4] = int(column_idx)

                    for playback_data in self.Header_index:
                        if playback_data[3] > playback_data[4]:
                            playback_data[4] = len(csv_data)

                    # print(self.Header_index)
                else:

                    mct_data = list(map(int, csv_data[self.Header_index[MCT_DELTAGEN_DATA][3]:
                                                      self.Header_index[MCT_DELTAGEN_DATA][4] + 1]))

                    col_data = list(map(int, csv_data[self.Header_index[SCTY_DELTAGEN_DATA][3]:
                                                      self.Header_index[SCTY_DELTAGEN_DATA][4] + 1]))

                    row_data = list(map(int, csv_data[self.Header_index[SCTX_DELTAGEN_DATA][3]:
                                                      self.Header_index[SCTX_DELTAGEN_DATA][4] + 1]))

                    mutual_raw.append(mct_data)
                    col_raw.append(col_data)
                    row_raw.append(row_data)
        f.close()

        # find the tx num and rx num using re
        Rx, Tx = re.findall(r"[\[](.*?)[\]]", header[self.Header_index[MCT_DELTAGEN_DATA][4]])
        self.row_num = int(Rx) + 1
        self.col_num = int(Tx) + 1

        self.mct_grid = np.array(mutual_raw).reshape([len(mutual_raw), self.row_num, self.col_num])
        # self.mct_grid = self.mct_grid[:300,:,:]
        # print(f"using {self.mct_grid.shape[0]} frames for evaluation!!")
        self.sct_row = np.array(row_raw)
        self.sct_col = np.array(col_raw)

    # *******************************************************************
    # ************    mutual grid field *********************************
    # *******************************************************************
    @property
    def mct_grid_max(self):
        return self.mct_grid.max(axis=0)

    @property
    def mct_grid_min(self):
        return self.mct_grid.min(axis=0)

    @property
    def mct_grid_mean(self):
        return self.mct_grid.mean(axis=0)

    @property
    def mct_grid_p2p(self):
        return self.mct_grid_max - self.mct_grid_min

    @property
    def mct_grid_rms(self):
        # alternative method
        # return np.sqrt(((self.mct_grid - self.mct_grid_mean) ** 2).mean(axis=0))
        return np.sqrt(np.var(self.mct_grid, axis=0))

    @property
    def mct_signal_position(self):
        n, y_node, x_node = np.unravel_index(self.mct_grid.argmax(), self.mct_grid.shape)
        return n, y_node, x_node

    @property
    def mct_signal_max(self):
        _, y_node, x_node = self.mct_signal_position
        return self.mct_grid_max[y_node][x_node]

    @property
    def mct_signal_min(self):
        _, y_node, x_node = self.mct_signal_position
        return self.mct_grid_min[y_node][x_node]

    @property
    def mct_signal_mean(self):
        _, y_node, x_node = self.mct_signal_position
        return self.mct_grid_mean[y_node][x_node]

    # *******************************************************************
    # ************************    self cap row field ********************
    # *******************************************************************

    @property
    def sct_row_max(self):
        return self.sct_row.max(axis=0)

    @property
    def sct_row_min(self):
        return self.sct_row.min(axis=0)

    @property
    def sct_row_mean(self):
        return self.sct_row.mean(axis=0)

    @property
    def sct_row_p2p(self):
        return self.sct_row_max - self.sct_row_min

    @property
    def sct_row_rms(self):
        # alternative method
        return np.sqrt(np.var(self.sct_row, axis=0))

    @property
    def sct_row_signal_position(self):
        n, y_node = np.unravel_index(self.sct_row.argmax(), self.sct_row.shape)
        return n, y_node

    @property
    def sct_row_signal_max(self):
        _, y_node = self.sct_row_signal_position
        return self.sct_row_max[y_node]

    @property
    def sct_row_signal_min(self):
        _, y_node = self.sct_row_signal_position
        return self.sct_row_min[y_node]

    @property
    def sct_row_signal_mean(self):
        _, y_node = self.sct_row_signal_position
        return self.sct_row_mean[y_node]

    # *******************************************************************
    # ************************  self cap columns field ******************
    # *******************************************************************

    @property
    def sct_col_max(self):
        return self.sct_col.max(axis=0)

    @property
    def sct_col_min(self):
        return self.sct_col.min(axis=0)

    @property
    def sct_col_mean(self):
        return self.sct_col.mean(axis=0)

    @property
    def sct_col_p2p(self):
        return self.sct_col_max - self.sct_col_min

    @property
    def sct_col_rms(self):
        # alternative method
        return np.sqrt(np.var(self.sct_col, axis=0))

    @property
    def sct_col_signal_position(self):
        n, y_node = np.unravel_index(self.sct_col.argmax(), self.sct_col.shape)
        return n, y_node

    @property
    def sct_col_signal_max(self):
        _, y_node = self.sct_col_signal_position
        return self.sct_col_max[y_node]

    @property
    def sct_col_signal_min(self):
        _, y_node = self.sct_col_signal_position
        return self.sct_col_min[y_node]

    @property
    def sct_col_signal_mean(self):
        _, y_node = self.sct_col_signal_position
        return self.sct_col_mean[y_node]

