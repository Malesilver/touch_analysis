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
from ETS_Dataframe import HEADER_ETS, ETS_Dataframe

file_dir = os.path.dirname(__file__)  # the directory that class "option" resides in
pd.set_option('display.max_columns', None)


class AnalyseData:
    def __init__(self,
                 no_touch_file_path: str = None,
                 touch_file_paths=None,
                 Header_index=None):
        if touch_file_paths is None:
            touch_file_paths = []
        self.pattern = os.path.basename(os.path.dirname(no_touch_file_path))
        self.standard_width_picture = [12.99, 8.49]
        self.Rows = None
        self.Columns = None

        self.NoTouchFrame: ETS_Dataframe = None
        self.TouchFrameSets: List[ETS_Dataframe] = []
        self.init_data_FrameSets(no_touch_file_path, touch_file_paths, Header_index)

        self.output_folder = os.path.join(os.path.dirname(no_touch_file_path), "output")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def init_data_FrameSets(self,
                            no_touch_file_path=None,
                            touch_file_paths=[],
                            Header_index=None):
        self.NoTouchFrame = ETS_Dataframe(file_path=no_touch_file_path, Header_index=Header_index)
        self.Rows = self.NoTouchFrame.row_num
        self.Columns = self.NoTouchFrame.col_num
        for touch_file_path in touch_file_paths:
            TouchFrame = ETS_Dataframe(file_path=touch_file_path, Header_index=Header_index)
            self.TouchFrameSets.append(TouchFrame)
            print(f"successfull load file {touch_file_path}")

    # ********************************************************
    # ***********MCT Field ***********************************
    # ********************************************************

    @property
    def mct_noise_p2p_grid(self):
        return self.NoTouchFrame.mct_grid_p2p

    @property
    def mct_noise_p2p_max(self):
        return self.NoTouchFrame.mct_grid_p2p.max()

    @property
    def mct_noise_p2p_mean(self):
        return self.NoTouchFrame.mct_grid_p2p.mean()

    @property
    def mct_noise_p2p_min(self):
        return self.NoTouchFrame.mct_grid_p2p.min()

    @property
    def all_touched_position(self):
        '''
        return the position of max value from all frames as touch node position
        :return: list node [[node1_x, node1_y] [node2_x node2_y] ...]
        '''
        ret = []
        for TouchFrame in self.TouchFrameSets:
            _, y_node, x_node = TouchFrame.mct_signal_position
            ret.append((y_node, x_node))
        return ret

    @property
    def all_mct_noise_p2p_notouch_node(self):
        '''
        The noise is taken from no touch peak-peak grid data at touched position
        :return: list node [node1 node2 ...]
        '''
        ret = []
        for TouchFrame in self.TouchFrameSets:
            _, y_node, x_node = TouchFrame.mct_signal_position
            ret.append(self.NoTouchFrame.mct_grid_p2p[y_node][x_node])
        return ret

    @property
    def all_mct_noise_p2p_touch_node(self):
        '''
        The noise is taken from touch peak-peak grid data at touched position
        :return: list node [node1 node2 ...]
        '''
        ret = []
        for TouchFrame in self.TouchFrameSets:
            _, y_node, x_node = TouchFrame.mct_signal_position
            ret.append(TouchFrame.mct_grid_p2p[y_node][x_node])
        return ret

    @property
    def all_mct_noise_rms_touch(self):
        """
        The noise is taken from rms noise (in touch raw data) grid data at touched position
        :return: list node [node1 node2 ...]
        """
        ret = []
        for TouchFrame in self.TouchFrameSets:
            _, y_node, x_node = TouchFrame.mct_signal_position
            ret.append(TouchFrame.mct_grid_rms[y_node][x_node])
        return ret

    @property
    def all_mct_signal_max(self):
        return [TouchData.mct_signal_max for TouchData in self.TouchFrameSets]

    @property
    def all_mct_signal_min(self):
        return [TouchData.mct_signal_min for TouchData in self.TouchFrameSets]

    @property
    def all_mct_signal_mean(self):
        return [TouchData.mct_signal_mean for TouchData in self.TouchFrameSets]

    @property
    def all_SminNppnotouchR(self):
        """
        (using in Huawei SNppR)
        Using min signal from touch raw data as signal, and peak-peak noise in no touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        """
        ret = []

        for TouchFrame in self.TouchFrameSets:
            _, y_node, x_node = TouchFrame.mct_signal_position
            SmaxNpptouchR = TouchFrame.mct_signal_min / self.NoTouchFrame.mct_grid_p2p[y_node][x_node]
            ret.append(SmaxNpptouchR)
        return ret

    @property
    def all_SmeanNppnotouchR(self):
        '''
        (using in Huawei SNppR)
        Using average signal from touch raw data as signal, and peak-peak noise in no touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        '''
        ret = []

        for TouchFrame in self.TouchFrameSets:
            _, y_node, x_node = TouchFrame.mct_signal_position
            SmaxNpptouchR = TouchFrame.mct_signal_mean / self.NoTouchFrame.mct_grid_p2p[y_node][x_node]
            ret.append(SmaxNpptouchR)
        return ret

    @property
    def all_SmaxNppnotouchR(self):
        '''
        (using in BOE SNppR) Using max signal from touch raw data as signal, and peak-peak noise in no touch raw data
        at touched node as noise :return: list of SNR [node1 node2 ...]
        '''
        ret = []

        for TouchFrame in self.TouchFrameSets:
            _, y_node, x_node = TouchFrame.mct_signal_position
            SmaxNpptouchR = TouchFrame.mct_signal_max / self.NoTouchFrame.mct_grid_p2p[y_node][x_node]
            ret.append(SmaxNpptouchR)
        return ret

    @property
    def all_SminNpptouchR(self):
        '''
        (using in Huawei quick test)
        Using min signal from touch raw data as signal, and peak-peak noise in touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        '''
        ret = []

        for TouchFrame in self.TouchFrameSets:
            _, y_node, x_node = TouchFrame.mct_signal_position
            SmaxNpptouchR = TouchFrame.mct_signal_min / TouchFrame.mct_grid_p2p[y_node][x_node]
            ret.append(SmaxNpptouchR)
        return ret

    @property
    def all_SmaxNppnotouchR_dB(self):

        return [20 * np.log10(val) for val in self.all_SmaxNppnotouchR]

    @property
    def all_SminNpptouchR_dB(self):
        '''
        (using in Huawei quick test)
        Using min signal from touch raw data as signal, and peak-peak noise in touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        '''

        return [20 * np.log10(val) for val in self.all_SminNpptouchR]

    @property
    def all_SminNppnotouchR_dB(self):
        '''
        (using in Huawei SNppR test)
        Using min signal from touch raw data as signal, and peak-peak noise in no touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        '''

        return [20 * np.log10(val) for val in self.all_SminNppnotouchR]

    @property
    def all_SmeanNppnotouchR_dB(self):
        '''
        (using in Huawei SNppR test)
        Using mean signal from touch raw data as signal, and peak-peak noise in no touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        '''

        return [20 * np.log10(val) for val in self.all_SmeanNppnotouchR]

    @property
    def all_SmaxNppfullscreenR(self):
        ret = []

        for TouchFrame in self.TouchFrameSets:
            SmaxNpptouchR = TouchFrame.mct_signal_max / self.NoTouchFrame.mct_grid_p2p.max()
            ret.append(SmaxNpptouchR)
        return ret

    @property
    def all_SmaxNppfullscreenR_dB(self):

        return [20 * np.log10(val) for val in self.all_SmaxNppfullscreenR]

    @property
    def all_SmeanNrmsR(self):
        ret = []
        for TouchFrame in self.TouchFrameSets:
            _, y_node, x_node = TouchFrame.mct_signal_position

            SmaxNrmsR = TouchFrame.mct_signal_mean / TouchFrame.mct_grid_rms[y_node][x_node]
            ret.append(SmaxNrmsR)
        return ret

    @property
    def all_SmeanNrmsR_dB(self):

        return [20 * np.log10(val) for val in self.all_SmeanNrmsR]

    # ********************************************************
    # ***********SCT ROW Field *******************************
    # ********************************************************

    @property
    def sct_row_noise_p2p_line(self):
        return self.NoTouchFrame.sct_row_p2p

    @property
    def sct_row_p2p_max(self):
        return self.NoTouchFrame.sct_row_p2p.max()

    @property
    def sct_row_p2p_mean(self):
        return self.NoTouchFrame.sct_row_p2p.mean()

    @property
    def sct_row_p2p_min(self):
        return self.NoTouchFrame.sct_row_p2p.min()

    # ********************************************************
    # ***********SCT COL Field *******************************
    # ********************************************************

    @property
    def sct_col_noise_p2p_line(self):
        return self.NoTouchFrame.sct_col_p2p

    @property
    def sct_col_p2p_max(self):
        return self.NoTouchFrame.sct_col_p2p.max()

    @property
    def sct_col_p2p_mean(self):
        return self.NoTouchFrame.sct_col_p2p.mean()

    @property
    def sct_col_p2p_min(self):
        return self.NoTouchFrame.sct_col_p2p.min()

    # ********************************************************
    # ***********SCT Results Field ([row],[col]) *******************************
    # ********************************************************

    @property
    def all_sct_touched_position(self):
        """
        return the position of max value from all frames as touch node position
        :return: list node [[node1_x, node1_y] [node2_x node2_y] ...]
        """
        ret_row = []
        ret_col = []

        for TouchFrame in self.TouchFrameSets:
            _, x_node = TouchFrame.sct_row_signal_position
            ret_row.append(x_node)

        for TouchFrame in self.TouchFrameSets:
            _, y_node = TouchFrame.sct_col_signal_position
            ret_col.append(y_node)

        return [ret_row, ret_col]

    @property
    def all_sct_noise_p2p_notouch_node(self):
        """
        The noise is taken from no touch peak-peak grid data at touched position
        :return: list node [node1 node2 ...]
        """
        ret_row = []
        ret_col = []

        for TouchFrame in self.TouchFrameSets:
            _, x_node = TouchFrame.sct_row_signal_position
            ret_row.append(self.NoTouchFrame.sct_row_p2p[x_node])

        for TouchFrame in self.TouchFrameSets:
            _, y_node = TouchFrame.sct_col_signal_position
            ret_col.append(self.NoTouchFrame.sct_col_p2p[y_node])

        return [ret_row, ret_col]

    @property
    def all_sct_noise_p2p_touch_node(self):
        """
        The noise is taken from touch peak-peak grid data at touched position
        :return: list node [node1 node2 ...]
        """
        ret_row = []
        ret_col = []

        for TouchFrame in self.TouchFrameSets:
            _, x_node = TouchFrame.sct_row_signal_position
            ret_row.append(TouchFrame.sct_row_p2p[x_node])

        for TouchFrame in self.TouchFrameSets:
            _, y_node = TouchFrame.sct_col_signal_position
            ret_col.append(TouchFrame.sct_col_p2p[y_node])

        return [ret_row, ret_col]

    @property
    def all_sct_noise_rms_touch(self):
        """
        The noise is taken from rms noise (in touch raw data) grid data at touched position
        :return: list node [node1 node2 ...]
        """
        ret_row = []
        ret_col = []

        for TouchFrame in self.TouchFrameSets:
            _, x_node = TouchFrame.sct_row_signal_position
            ret_row.append(TouchFrame.sct_row_rms[x_node])

        for TouchFrame in self.TouchFrameSets:
            _, y_node = TouchFrame.sct_col_signal_position
            ret_col.append(TouchFrame.sct_col_rms[y_node])

        return [ret_row, ret_col]

    @property
    def all_sct_signal_max(self):
        ret_row = [TouchData.sct_row_signal_max for TouchData in self.TouchFrameSets]
        ret_col = [TouchData.sct_col_signal_max for TouchData in self.TouchFrameSets]
        return [ret_row, ret_col]

    @property
    def all_sct_signal_min(self):
        ret_row = [TouchData.sct_row_signal_min for TouchData in self.TouchFrameSets]
        ret_col = [TouchData.sct_col_signal_min for TouchData in self.TouchFrameSets]
        return [ret_row, ret_col]

    @property
    def all_sct_signal_mean(self):
        ret_row = [TouchData.sct_row_signal_mean for TouchData in self.TouchFrameSets]
        ret_col = [TouchData.sct_col_signal_mean for TouchData in self.TouchFrameSets]
        return [ret_row, ret_col]

    @property
    def all_sct_SminNppnotouchR(self):
        """
        (using in Huawei SNppR)
        Using min signal from touch raw data as signal, and peak-peak noise in no touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        """
        ret_row = []
        ret_col = []

        for TouchFrame in self.TouchFrameSets:
            _, x_node = TouchFrame.sct_row_signal_position
            SminNppnotouchR = TouchFrame.sct_row_signal_min / self.NoTouchFrame.sct_row_p2p[x_node]
            ret_row.append(SminNppnotouchR)

        for TouchFrame in self.TouchFrameSets:
            _, y_node = TouchFrame.sct_col_signal_position
            SminNppnotouchR = TouchFrame.sct_col_signal_min / self.NoTouchFrame.sct_col_p2p[y_node]
            ret_col.append(SminNppnotouchR)

        return [ret_row, ret_col]

    @property
    def all_sct_SmeanNppnotouchR(self):
        '''
        (using in Huawei SNppR)
        Using average signal from touch raw data as signal, and peak-peak noise in no touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        '''

        ret_row = []
        ret_col = []
        # calculate sct row SmeanNppnotouchR
        for TouchFrame in self.TouchFrameSets:
            _, x_node = TouchFrame.sct_row_signal_position
            SmeanNppnotouchR = TouchFrame.sct_row_signal_mean / self.NoTouchFrame.sct_row_p2p[x_node]
            ret_row.append(SmeanNppnotouchR)

        # calculate sct col SmeanNppnotouchR
        for TouchFrame in self.TouchFrameSets:
            _, y_node = TouchFrame.sct_col_signal_position
            SmeanNppnotouchR = TouchFrame.sct_col_signal_mean / self.NoTouchFrame.sct_col_p2p[y_node]
            ret_col.append(SmeanNppnotouchR)

        return [ret_row, ret_col]

    @property
    def all_sct_SmaxNppnotouchR(self):
        '''
        (using in BOE SNppR) Using max signal from touch raw data as signal, and peak-peak noise in no touch raw data
        at touched node as noise :return: list of SNR [node1 node2 ...]
        '''
        ret_row = []
        ret_col = []

        for TouchFrame in self.TouchFrameSets:
            _, x_node = TouchFrame.sct_row_signal_position
            SmaxNppnotouchR = TouchFrame.sct_row_signal_max / self.NoTouchFrame.sct_row_p2p[x_node]
            ret_row.append(SmaxNppnotouchR)

        for TouchFrame in self.TouchFrameSets:
            _, y_node = TouchFrame.sct_col_signal_position
            SmaxNppnotouchR = TouchFrame.sct_col_signal_max / self.NoTouchFrame.sct_col_p2p[y_node]
            ret_col.append(SmaxNppnotouchR)

        return [ret_row, ret_col]

    @property
    def all_sct_SminNpptouchR(self):
        '''
        (using in Huawei quick test)
        Using min signal from touch raw data as signal, and peak-peak noise in touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        '''
        ret_row = []
        ret_col = []

        for TouchFrame in self.TouchFrameSets:
            _, x_node = TouchFrame.sct_row_signal_position
            SminNpptouchR = TouchFrame.sct_row_signal_min / TouchFrame.sct_row_p2p[x_node]
            ret_row.append(SminNpptouchR)

        for TouchFrame in self.TouchFrameSets:
            _, y_node = TouchFrame.sct_col_signal_position
            SminNpptouchR = TouchFrame.sct_col_signal_min / TouchFrame.sct_col_p2p[y_node]
            ret_col.append(SminNpptouchR)

        return [ret_row, ret_col]

    @property
    def all_sct_SmaxNppnotouchR_dB(self):
        ret_row = [20 * np.log10(val) for val in self.all_sct_SmaxNppnotouchR[0]]
        ret_col = [20 * np.log10(val) for val in self.all_sct_SmaxNppnotouchR[1]]
        return [ret_row, ret_col]

    @property
    def all_sct_SminNpptouchR_dB(self):
        """
        (using in Huawei quick test)
        Using min signal from touch raw data as signal, and peak-peak noise in touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        """
        ret_row = [20 * np.log10(val) for val in self.all_sct_SminNpptouchR[0]]
        ret_col = [20 * np.log10(val) for val in self.all_sct_SminNpptouchR[1]]
        return [ret_row, ret_col]

    @property
    def all_sct_SminNppnotouchR_dB(self):
        """
        (using in Huawei SNppR test)
        Using min signal from touch raw data as signal, and peak-peak noise in no touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        """

        ret_row = [20 * np.log10(val) for val in self.all_sct_SminNppnotouchR[0]]
        ret_col = [20 * np.log10(val) for val in self.all_sct_SminNppnotouchR[1]]
        return [ret_row, ret_col]

    @property
    def all_sct_SmeanNppnotouchR_dB(self):
        """
        (using in Huawei SNppR test)
        Using mean signal from touch raw data as signal, and peak-peak noise in no touch raw data at touched node as noise
        :return: list of SNR [node1 node2 ...]
        """
        ret_row = [20 * np.log10(val) for val in self.all_sct_SmeanNppnotouchR[0]]
        ret_col = [20 * np.log10(val) for val in self.all_sct_SmeanNppnotouchR[1]]
        return [ret_row, ret_col]

    @property
    def all_sct_SmaxNppfullscreenR(self):

        ret_row = []
        ret_col = []

        for TouchFrame in self.TouchFrameSets:
            SmaxNppfullscreenR = TouchFrame.sct_row_signal_max / self.NoTouchFrame.sct_row_p2p.max()
            ret_row.append(SmaxNppfullscreenR)

        for TouchFrame in self.TouchFrameSets:
            SmaxNppfullscreenR = TouchFrame.sct_col_signal_max / self.NoTouchFrame.sct_col_p2p.max()
            ret_col.append(SmaxNppfullscreenR)

        return [ret_row, ret_col]

    @property
    def all_sct_SmaxNppfullscreenR_dB(self):

        ret_row = [20 * np.log10(val) for val in self.all_sct_SmaxNppfullscreenR[0]]
        ret_col = [20 * np.log10(val) for val in self.all_sct_SmaxNppfullscreenR[1]]
        return [ret_row, ret_col]

    @property
    def all_sct_SmeanNrmsR(self):

        ret_row = []
        ret_col = []

        for TouchFrame in self.TouchFrameSets:
            _, x_node = TouchFrame.sct_row_signal_position
            SmeanNrmsR = TouchFrame.sct_row_signal_mean / TouchFrame.sct_row_rms[x_node]
            ret_row.append(SmeanNrmsR)

        for TouchFrame in self.TouchFrameSets:
            _, y_node = TouchFrame.sct_col_signal_position
            SmeanNrmsR = TouchFrame.sct_col_signal_mean / TouchFrame.sct_col_rms[y_node]
            ret_col.append(SmeanNrmsR)

        return [ret_row, ret_col]

    @property
    def all_sct_SmeanNrmsR_dB(self):

        ret_row = [20 * np.log10(val) for val in self.all_sct_SmeanNrmsR[0]]
        ret_col = [20 * np.log10(val) for val in self.all_sct_SmeanNrmsR[1]]
        return [ret_row, ret_col]

    def BOE_snr_summary(self):
        ret = {"Vendor": "BOE"}
        if self.NoTouchFrame.mct_grid is not None:
            min_SmaxNppfullscreenR_dB = min(self.all_SmaxNppfullscreenR_dB)
            min_SmaxNppfullscreenR_dB_index = self.all_SmaxNppfullscreenR_dB.index(min(self.all_SmaxNppfullscreenR_dB))
            min_SmeanNrmsR_dB = min(self.all_SmeanNrmsR_dB)
            min_SmeanNrmsR_dB_index = self.all_SmeanNrmsR_dB.index(min(self.all_SmeanNrmsR_dB))
            mct_ret = {
                "snr_summary": {
                    "touched node": self.all_touched_position,
                    "noise_p2p_fullscreen": [self.mct_noise_p2p_max] * len(self.TouchFrameSets),
                    "noise_p2p_notouch": self.all_mct_noise_p2p_notouch_node,
                    "noise_rms_touch": self.all_mct_noise_rms_touch,
                    "signal_max": self.all_mct_signal_max,
                    "signal_mean": self.all_mct_signal_mean,
                    "SmaxNppmotouchR": self.all_SmaxNppnotouchR,
                    "SmaxNppnotouchR_dB": self.all_SmaxNppnotouchR_dB,
                    "SmaxNppfullscreenR": self.all_SmaxNppfullscreenR,
                    "SmaxNppfullscreenR_dB": self.all_SmaxNppfullscreenR_dB,
                    "SmaxNrmsR": self.all_SmeanNrmsR,
                    "SmeanNrmsR_dB": self.all_SmeanNrmsR_dB
                },
                "final_results": {
                    "min_SmaxNppfullscreenR_dB": "{:.2f}".format(min_SmaxNppfullscreenR_dB),
                    "Position_P2P": f"Touch {min_SmaxNppfullscreenR_dB_index + 1}",
                    "min_SmeanNrmsR_dB": "{:.2f}".format(min_SmeanNrmsR_dB),
                    "Position_RMS": f"Touch {min_SmeanNrmsR_dB_index + 1}"
                }
            }
            ret["mct_summary"] = mct_ret

        if self.NoTouchFrame.sct_row is not None:
            min_sct_row_SmaxNppfullscreenR_dB = min(self.all_sct_SmaxNppfullscreenR_dB[0])
            min_sct_row_SmaxNppfullscreenR_dB_index = self.all_sct_SmaxNppfullscreenR_dB[0].index(
                min(self.all_sct_SmaxNppfullscreenR_dB[0]))
            min_sct_row_SmeanNrmsR_dB = min(self.all_sct_SmeanNrmsR_dB[0])
            min_sct_row_SmeanNrmsR_dB_index = self.all_sct_SmeanNrmsR_dB[0].index(min(self.all_sct_SmeanNrmsR_dB[0]))

            sct_row_ret = {
                "snr_sct_row_summary": {
                    "touched node": self.all_sct_touched_position[0],
                    "noise_p2p_fullscreen": [self.sct_row_p2p_max] * len(self.TouchFrameSets),
                    "noise_p2p_notouch": self.all_sct_noise_p2p_notouch_node[0],
                    "noise_rms_touch": self.all_sct_noise_rms_touch[0],
                    "signal_max": self.all_sct_signal_max[0],
                    "signal_mean": self.all_sct_signal_mean[0],
                    "SmaxNppmotouchR": self.all_sct_SmaxNppnotouchR[0],
                    "SmaxNppnotouchR_dB": self.all_sct_SmaxNppnotouchR_dB[0],
                    "SmaxNppfullscreenR": self.all_sct_SmaxNppfullscreenR[0],
                    "SmaxNppfullscreenR_dB": self.all_sct_SmaxNppfullscreenR_dB[0],
                    "SmaxNrmsR": self.all_sct_SmeanNrmsR[0],
                    "SmeanNrmsR_dB": self.all_sct_SmeanNrmsR_dB[0]
                },
                "final_results": {
                    "min_sct_row_SmaxNppfullscreenR_dB": "{:.2f}".format(min_sct_row_SmaxNppfullscreenR_dB),
                    "Position_P2P": f"Touch {min_sct_row_SmaxNppfullscreenR_dB_index + 1}",
                    "min_sct_row_SmeanNrmsR_dB": "{:.2f}".format(min_sct_row_SmeanNrmsR_dB),
                    "Position_RMS": f"Touch {min_sct_row_SmeanNrmsR_dB_index + 1}"
                }
            }
            ret["sct_row_summary"] = sct_row_ret

        if self.NoTouchFrame.sct_col is not None:
            min_sct_col_SmaxNppfullscreenR_dB = min(self.all_sct_SmaxNppfullscreenR_dB[1])
            min_sct_col_SmaxNppfullscreenR_dB_index = self.all_sct_SmaxNppfullscreenR_dB[1].index(
                min(self.all_sct_SmaxNppfullscreenR_dB[1]))
            min_sct_col_SmeanNrmsR_dB = min(self.all_sct_SmeanNrmsR_dB[1])
            min_sct_col_SmeanNrmsR_dB_index = self.all_sct_SmeanNrmsR_dB[1].index(min(self.all_sct_SmeanNrmsR_dB[1]))
            sct_col_ret = {
                "snr_sct_col_summary": {
                    "touched node": self.all_sct_touched_position[1],
                    "noise_p2p_fullscreen": [self.sct_col_p2p_max] * len(self.TouchFrameSets),
                    "noise_p2p_notouch": self.all_sct_noise_p2p_notouch_node[1],
                    "noise_rms_touch": self.all_sct_noise_rms_touch[1],
                    "signal_max": self.all_sct_signal_max[1],
                    "signal_mean": self.all_sct_signal_mean[1],
                    "SmaxNppmotouchR": self.all_sct_SmaxNppnotouchR[1],
                    "SmaxNppnotouchR_dB": self.all_sct_SmaxNppnotouchR_dB[1],
                    "SmaxNppfullscreenR": self.all_sct_SmaxNppfullscreenR[1],
                    "SmaxNppfullscreenR_dB": self.all_sct_SmaxNppfullscreenR_dB[1],
                    "SmaxNrmsR": self.all_sct_SmeanNrmsR[1],
                    "SmeanNrmsR_dB": self.all_sct_SmeanNrmsR_dB[1]
                },
                "final_results": {
                    "min_sct_col_SmaxNppfullscreenR_dB": "{:.2f}".format(min_sct_col_SmaxNppfullscreenR_dB),
                    "Position_P2P": f"Touch {min_sct_col_SmaxNppfullscreenR_dB_index + 1}",
                    "min_sct_col_SmeanNrmsR_dB": "{:.2f}".format(min_sct_col_SmeanNrmsR_dB),
                    "Position_RMS": f"Touch {min_sct_col_SmeanNrmsR_dB_index + 1}"
                }
            }
            ret["sct_col_summary"] = sct_col_ret

        return ret

    def HW_quick_snr_summary(self):
        '''
        huawei quick test only need touch raw data!!
        only peak-peak snr is calculated!!
        :return:
        '''
        ret = {"Vendor": "Huawei_quick"}

        min_SminNpptouch_dB = min(self.all_SminNpptouchR_dB)
        min_SminNpptouch_dB_index = self.all_SminNpptouchR_dB.index(min(self.all_SminNpptouchR_dB))
        mct_ret = {
            "snr_summary": {
                "touched node": self.all_touched_position,
                "noise_p2p_touch": self.all_mct_noise_p2p_touch_node,
                "signal_min": self.all_mct_signal_min,
                "SminNpptouchR": self.all_SminNpptouchR,
                "SminNpptouchR_dB": self.all_SminNpptouchR_dB,
            },
            "final_results": {
                "min_SminNpptouch_dB": "{:.2f}".format(min_SminNpptouch_dB),
                "min_SminNpptouch_dB_index": f"Touch {min_SminNpptouch_dB_index + 1}",
            }
        }
        ret["mct_summary"] = mct_ret
        return ret

    def HW_thp_afe_snr_summary(self):
        '''
        huawei thp afe test using no touch data as the source of noise!!
        only peak-peak snr is calculated!!
        :return:
        '''
        ret = {"Vendor": "Huawei_quick"}

        min_SminNppnotouch_dB = min(self.all_SminNppnotouchR_dB)
        min_SminNppnotouch_dB_index = self.all_SminNppnotouchR_dB.index(min(self.all_SminNppnotouchR_dB))
        min_SmeanNppnotouch_dB = min(self.all_SmeanNppnotouchR_dB)
        min_SmeanNppnotouch_dB_index = self.all_SmeanNppnotouchR_dB.index(min(self.all_SmeanNppnotouchR_dB))
        mct_ret = {
            "snr_summary": {
                "touched node": self.all_touched_position,
                "noise_p2p_notouch": self.all_mct_noise_p2p_notouch_node,
                "signal_min": self.all_mct_signal_min,
                "signal_mean": self.all_mct_signal_mean,
                "SminNppnotouchR": self.all_SminNppnotouchR,
                "SminNppnotouchR_dB": self.all_SminNppnotouchR_dB,
                "SmeanNppnotouchR": self.all_SmeanNppnotouchR,
                "SmeanNppnotouchR_dB": self.all_SmeanNppnotouchR_dB,
            },
            "final_results": {
                "min_SminNppnotouch_dB": "{:.2f}".format(min_SminNppnotouch_dB),
                "min_SminNppnotouch_dB_index": f"Touch {min_SminNppnotouch_dB_index + 1}",
                "min_SmeanNppnotouch_dB": "{:.2f}".format(min_SmeanNppnotouch_dB),
                "min_SmeanNppnotouch_dB_index": f"Touch {min_SmeanNppnotouch_dB_index + 1}",
            }
        }
        ret["mct_summary"] = mct_ret
        return ret

    def write_out_csv(self, result_dict):

        row = ["Touch {}".format(idx + 1) for idx in range(len(self.TouchFrameSets))]
        output_path = os.path.join(self.output_folder, result_dict["Vendor"] + "_" + self.pattern + "_output_info.csv")

        with open(output_path, 'w', newline='') as f:

            if self.NoTouchFrame.mct_grid is not None:
                data_out = pd.DataFrame(index=row,
                                        data=result_dict["mct_summary"]["snr_summary"])
                # print(data_out)
                data_out = data_out.round(2)
                # print(data_out)
                csv_write = csv.writer(f)
                csv_write.writerow(["MCT Summary"])
                data_out.to_csv(f)

            if self.NoTouchFrame.sct_row is not None:
                data_out = pd.DataFrame(index=row,
                                        data=result_dict["sct_row_summary"]["snr_sct_row_summary"])
                data_out = data_out.round(2)
                csv_write = csv.writer(f)
                csv_write.writerow("\n")
                csv_write.writerow(["SCT Row Summary"])

                data_out.to_csv(f)

            if self.NoTouchFrame.sct_col is not None:
                data_out = pd.DataFrame(index=row,
                                        data=result_dict["sct_col_summary"]["snr_sct_col_summary"])
                data_out = data_out.round(2)
                csv_write = csv.writer(f)
                csv_write.writerow("\n")
                csv_write.writerow(["SCT Column Summary"])
                data_out.to_csv(f)

        if os.path.exists(output_path):
            with open(output_path, 'a+', newline='') as f:
                csv_write = csv.writer(f)

                if self.NoTouchFrame.mct_grid is not None:
                    csv_write.writerow("\n")

                    csv_write.writerow(["Final Result MCT:"])
                    for x, y in zip(list(result_dict["mct_summary"]["final_results"].keys()),
                                    list(result_dict["mct_summary"]["final_results"].values())):
                        csv_write.writerow([x, y])

                if self.NoTouchFrame.sct_row is not None:
                    csv_write.writerow("\n")

                    csv_write.writerow(["Final Result SCT Row:"])
                    for x, y in zip(list(result_dict["sct_row_summary"]["final_results"].keys()),
                                    list(result_dict["sct_row_summary"]["final_results"].values())):
                        csv_write.writerow([x, y])

                if self.NoTouchFrame.sct_col is not None:
                    csv_write.writerow("\n")

                    csv_write.writerow(["Final Result SCT Col:"])
                    for x, y in zip(list(result_dict["sct_col_summary"]["final_results"].keys()),
                                    list(result_dict["sct_col_summary"]["final_results"].values())):
                        csv_write.writerow([x, y])

    def write_out_decode_mct_csv(self):

        # write out No Touch grid mct raw data
        notouch_out_path = os.path.join(self.output_folder, self.pattern + "_mct_grid_rawdata_NoTouch.csv")
        with open(notouch_out_path, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            all_mct_data = self.NoTouchFrame.mct_grid
            for idx, mct_grid in enumerate(all_mct_data):
                # write a row to the csv file
                writer.writerow(["Frame {}:".format(idx + 1)])

                for line in mct_grid:
                    li = list(line)
                    writer.writerow(li)
                writer.writerow("\n")
        f.close()

        for idx, TouchFrame in enumerate(self.TouchFrameSets):
            touch_out_path = os.path.join(self.output_folder,
                                          self.pattern + "_mct_grid_rawdata_Touch_{}.csv".format(int(idx)))
            with open(touch_out_path, 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                all_mct_data = TouchFrame.mct_grid
                for idx, mct_grid in enumerate(all_mct_data):
                    # write a row to the csv file
                    writer.writerow(["Frame {}:".format(idx + 1)])

                    for line in mct_grid:
                        li = list(line)
                        writer.writerow(li)
                    writer.writerow("\n")
            f.close()

    def plot_mct_noise_rms(self):
        ret = []
        touched_node_list = self.all_touched_position
        for idx, TouchFrame in enumerate(self.TouchFrameSets):
            ynode, xnode = touched_node_list[idx]
            grid_Data = TouchFrame.mct_grid_rms
            fig = plt.figure(figsize=self.standard_width_picture, dpi=110)
            ax = sns.heatmap(data=grid_Data, annot=True, fmt='.0f', vmin=0)
            ax.set_ylabel('Row')
            ax.set_xlabel('Column')
            plt.title('Grid RMS Noise without Touch [%d , %d]\n touched node is [%d,%d] -> RMS Noise is %.0f' % (
                self.NoTouchFrame.row_num, self.NoTouchFrame.col_num, ynode, xnode, grid_Data[ynode][xnode])
                      )
            # annotate touched position
            self.annotate_grid_figure(ax=ax, ynode=ynode, xnode=xnode, Text=f"Touch {idx + 1}")

            plt.tight_layout()
            fig.savefig(os.path.join(self.output_folder, "Figure_MCT_rms_noise_touch_{}.png").format(idx))
            print("Successfully generate figure {}!!!!!".format("Figure_MCT_rms_noise_touch_{}.png").format(idx))
            ret.append(fig)
        return ret

    def plot_mct_noise_p2p_annotated(self):
        grid_Data = self.NoTouchFrame.mct_grid_p2p
        fig = plt.figure(figsize=self.standard_width_picture, dpi=110)
        ax = sns.heatmap(data=grid_Data, annot=True, fmt='.0f', vmin=0, vmax=1000)
        ax.set_ylabel('Row')
        ax.set_xlabel('Column')
        plt.title('Grid Peak-Peak Noise without Touch [%d , %d]\n Mean=%.0f; Min=%.0f; Max=%.0f' % (
            self.NoTouchFrame.row_num, self.NoTouchFrame.col_num, grid_Data.mean(), grid_Data.min(), grid_Data.max())
                  )

        for idx, (ynode, xnode) in enumerate(self.all_touched_position):
            self.annotate_grid_figure(ax=ax, ynode=ynode, xnode=xnode, Text=f"Touch {idx + 1}")

        plt.tight_layout()
        fig.savefig(os.path.join(self.output_folder, "Figure_MCT_p2p_noise_annotated.png"))
        print("Successfully generate figure {}!!!!!".format("Figure_MCT_p2p_noise_annotated.png"))
        return fig

    def plot_mct_noise_p2p(self):
        grid_Data = self.NoTouchFrame.mct_grid_p2p
        fig = plt.figure(figsize=self.standard_width_picture, dpi=110)
        ax = sns.heatmap(data=grid_Data, annot=True, fmt='.0f', vmin=0, vmax=1000)
        ax.set_ylabel('Row')
        ax.set_xlabel('Column')
        plt.title('Grid Peak-Peak Noise without Touch [%d , %d]\n Mean=%.0f; Min=%.0f; Max=%.0f' % (
            self.NoTouchFrame.row_num, self.NoTouchFrame.col_num, grid_Data.mean(), grid_Data.min(), grid_Data.max())
                  )

        plt.tight_layout()
        fig.savefig(os.path.join(self.output_folder, "Figure_MCT_p2p_noise.png"))
        return fig

    def plot_touch_signal_all(self):
        touch_data = []
        signal_list = []
        for TouchFrame in self.TouchFrameSets:
            signal = TouchFrame.mct_signal_max
            touch_grid = TouchFrame.mct_grid_mean
            _, ynode, xnode = TouchFrame.mct_signal_position
            touch_grid[ynode][xnode] = signal
            touch_data.append(touch_grid)
            signal_list.append(signal)

        touch_data = np.array(touch_data)
        signal_array = np.array(signal_list)
        grid_Data = touch_data.max(axis=0)
        fig = plt.figure(figsize=self.standard_width_picture, dpi=110)
        ax = sns.heatmap(data=grid_Data, annot=True, fmt='.0f', vmin=0, vmax=1000)
        ax.set_ylabel('Row')
        ax.set_xlabel('Column')
        plt.title('Grid signal with Touch [%d , %d] \nMean=%.0f; Min=%.0f; Max=%.0f' % (
            self.NoTouchFrame.row_num, self.NoTouchFrame.col_num,
            signal_array.mean(), signal_array.min(), signal_array.max())
                  )

        plt.tight_layout()

        fig.savefig(os.path.join(self.output_folder, "Figure_MCT_all_signals.png"))
        return fig

    def annotate_grid_figure(self, ax, ynode, xnode, Text, boxcoler="cyan", arrowcolor="cyan"):

        ax.add_patch(patches.Rectangle((xnode, ynode), 1, 1, fill=False,
                                       facecolor=None, edgecolor=boxcoler, linewidth=4.0))
        if xnode < (self.Columns * 2) / 3:
            # exit arrow on the right side
            if ynode < (self.Rows * 2) / 3:
                # exit arrow on the bottom
                arrow_x, arrow_y = xnode + 1, ynode + 1
                textbox_x, textbox_y = xnode + 1.5, ynode + 3

            else:
                # exit arrow on the top
                arrow_x, arrow_y = xnode + 1, ynode
                textbox_x, textbox_y = xnode + 1.5, ynode - 2
        else:
            # exit on the left
            # exit arrow on the right side
            if ynode < (self.Rows * 2) / 3:
                # exit arrow on the bottom
                arrow_x, arrow_y = xnode, ynode + 1
                textbox_x, textbox_y = xnode - 1.5, ynode + 3

            else:
                # exit arrow on the top
                arrow_x, arrow_y = xnode, ynode
                textbox_x, textbox_y = xnode - 1.5, ynode - 2

        ax.annotate(Text,
                    xy=(arrow_x, arrow_y), xycoords='data',
                    xytext=(textbox_x, textbox_y), textcoords='data',
                    bbox=dict(boxstyle="round", fc=boxcoler, alpha=0.65),
                    arrowprops=dict(arrowstyle="->", edgecolor=arrowcolor,
                                    connectionstyle="angle,angleA=90,angleB=0,rad=10"))

    # def phase_analysis(self):
    #     ref_sig = self.NoTouchFrame.mct_grid_mean
    #     for TouchFrame in self.TouchFrameSets:
    #         touch_sig = TouchFrame.mct_grid_mean
    #         print(calc_mut_phase_compensation(touch_sig, ref_sig))


def get_touched_num(folder, prefix):
    """

    :param folder: raw data folder
    :return: index number list. i.e [wi1.csv,w2.csv,w5.csv] -> [1,2,5]
    """
    files = os.listdir(folder)
    ret = []

    for file in files:
        if re.search('{}(\d+)'.format(prefix), file):
            ret.append(re.search('{}(\d+)'.format(prefix), file).group(1))

    return ret


def write_out_final_result_csv(folder, data):
    # write out No Touch grid mct raw data
    out_path = os.path.join(folder, "result_summary.csv")
    with open(out_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(["pattern (fullscreen)", "SNppR MCT", "SNrmsR MCT", "SNppR SCT Row",
                         "SNrmsR SCT Row","SNppR SCT Col", "SNrmsR SCT Col"])
        for idx, line in enumerate(data):
            li = list(line)
            writer.writerow(li)

    f.close()


class SNRToolingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SNR calculation tool options")

        # PATH of dataset
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="folder path to the raw data",
                                 default=os.path.join(file_dir, "BOE_noise_test"))

        self.parser.add_argument('--pattern_folder',
                                 nargs='+',
                                 default=['white'],
                                 help='<Required> Set flag',
                                 required=True)

        # todo now output path is hard coded
        # self.parser.add_argument("--log_dir",
        #                          type=str,
        #                          help="log directory",
        #                          default=os.path.join(os.path.expanduser("~"), "tmp"))

        # self.parser.add_argument("--report_vendor",
        #                          type=str,
        #                          help="select right report vendor",
        #                          default="BOE",
        #                          choices=["BOE", "Huawei", "Visionox", "CSOT"])

        self.parser.add_argument('--report_vendor',
                                 action='append',
                                 help='<Required> Set flag',
                                 default=[],
                                 required=True)
        self.parser.add_argument("--prefix_notouch",
                                 type=str,
                                 help="pattern type",
                                 default="w",
                                 )

        self.parser.add_argument("--prefix_touch",
                                 type=str,
                                 help="pattern type",
                                 default="w",
                                 )

        self.parser.add_argument("--plot_noise_p2p",
                                 help="plot no touch p2p noise heatmap",
                                 action="store_true")
        self.parser.add_argument("--plot_noise_rms",
                                 help="plot no touch rms noise heatmap",
                                 action="store_true")

        self.parser.add_argument("--plot_all_touch_sigal",
                                 help="plot all touch signal in one heatmap",
                                 action="store_true")

        self.parser.add_argument("--plot_noise_p2p_annotated",
                                 help="plot no touch p2p noise heatmap with touch num annotation",
                                 action="store_true")

        self.parser.add_argument("--log_grid_rawdata",
                                 help="convert mct rawdata into grid foramt",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


if __name__ == '__main__':
    # load configuration options
    options = SNRToolingOptions()
    opts = options.parse()
    final_results = []

    for pattern in opts.pattern_folder:
        # # rawdata folder
        # pattern_folder = opts.pattern_folder
        # print(pattern_folder)
        # pattern = os.path.basename(pattern_folder)

        # modify rawdata paths: path format is **.edl.csv, i.e:
        # notouch path "wo.edl.csv" -> prefix_notouch = "wo"
        # touch path "wi5.edl.csv" -> prefix_touch = "wi"
        prefix_notouch = opts.prefix_notouch
        prefix_touch = opts.prefix_touch

        if os.path.exists(os.path.join(opts.dataset, pattern, "{}.edl.csv".format(prefix_notouch))):
            notouch_data_path = os.path.join(opts.dataset, pattern, "{}.edl.csv".format(prefix_notouch))
            touch_path = os.path.join(opts.dataset, pattern, prefix_touch + "{}.edl.csv")
        else:
            notouch_data_path = os.path.join(opts.dataset, pattern, "{}.csv".format(prefix_notouch))
            touch_path = os.path.join(opts.dataset, pattern, prefix_touch + "{}.csv")

        touch_list = get_touched_num(os.path.join(opts.dataset, pattern), prefix_touch)

        # match touch raw data file
        touch_data_path_list = [touch_path.format(i) for i in touch_list]

        # AnalyseData is main class for snr analysis
        DataAnalyse = AnalyseData(no_touch_file_path=notouch_data_path,
                                  touch_file_paths=touch_data_path_list,
                                  Header_index=HEADER_ETS)

        # print(pd.DataFrame(DataAnalyse.BOE_snr_summary()))

        # select vendor for different report
        if "BOE" in opts.report_vendor:
            BOE_ret = DataAnalyse.BOE_snr_summary()
            DataAnalyse.write_out_csv(BOE_ret)
            # "min_SmaxNppfullscreenR_dB": "{:.2f}".format(min_SmaxNppfullscreenR_dB),
            # "Position_P2P": f"Touch {min_SmaxNppfullscreenR_dB_index + 1}",
            # "min_SmeanNrmsR_dB": "{:.2f}".format(min_SmeanNrmsR_dB),
            # "Position_RMS": f"Touch {min_SmeanNrmsR_dB_index + 1}"
            tmp_res = [pattern]
            if BOE_ret.get("mct_summary", None) is not None:
                tmp_res.extend([ BOE_ret["mct_summary"]["final_results"]["min_SmaxNppfullscreenR_dB"],
                                BOE_ret["mct_summary"]["final_results"]["min_SmeanNrmsR_dB"]])
            else:
                tmp_res.extend(["NaN", "NaN"])

            if BOE_ret.get("sct_row_summary", None) is not None:
                tmp_res.extend([BOE_ret["sct_row_summary"]["final_results"]["min_sct_row_SmaxNppfullscreenR_dB"],
                                BOE_ret["sct_row_summary"]["final_results"]["min_sct_row_SmeanNrmsR_dB"]])
            else:
                tmp_res.extend(["NaN", "NaN"])

            if BOE_ret.get("sct_col_summary", None) is not None:
                tmp_res.extend([BOE_ret["sct_col_summary"]["final_results"]["min_sct_col_SmaxNppfullscreenR_dB"],
                                BOE_ret["sct_col_summary"]["final_results"]["min_sct_col_SmeanNrmsR_dB"]])
            else:
                tmp_res.extend(["NaN", "NaN"])
            final_results.append(tmp_res)
        if "Huawei_quick" in opts.report_vendor:
            DataAnalyse.write_out_csv(DataAnalyse.HW_quick_snr_summary())

        if "Huawei_thp_afe" in opts.report_vendor:
            DataAnalyse.write_out_csv(DataAnalyse.HW_thp_afe_snr_summary())

        # convert mct rawdata into grid foramt
        if opts.log_grid_rawdata:
            DataAnalyse.write_out_decode_mct_csv()

        # plot no touch p2p noise heatmap
        if opts.plot_noise_p2p:
            DataAnalyse.plot_mct_noise_p2p()

        # plot no touch rms noise heatmap
        if opts.plot_noise_rms:
            DataAnalyse.plot_mct_noise_rms()

        # plot all touch signal in one heatmap
        if opts.plot_all_touch_sigal:
            DataAnalyse.plot_touch_signal_all()

        if opts.plot_noise_p2p_annotated:
            DataAnalyse.plot_mct_noise_p2p_annotated()

        print(f"Already successful finish {pattern} !!!!!!!!!!!!!!")

    write_out_final_result_csv(opts.dataset, final_results)
