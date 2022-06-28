"""Module providing phase compensation functions """

import math
from typing import Callable
import numpy as np


def fmax_search(fun: Callable,
                cparam: complex,
                left: float,
                right: float,
                tol: float = 2 * math.pi / 100,
                max_iter: int = 10) -> float:
    """
    Generic binary search function to find the value of x which maximises the function f(x,y) where y is an additional
    constant parameter. Assumes that there is a single unique maximum within the specified range.
    :param fun: function f(x,y) to maximise, where x is in the range left to right, and y is cparam.
    :param cparam: complex additional parameter to function
    :param left: float scalar sets the lower limit of the search range.
    :param right: float scalar sets the upper limit of the search range.
    :param tol: float scale set the tolerance of the search, and precision of the result.
    :param max_iter: integer scaler to set the maximum number of binary search iterations.
    :return: float, value of x which maximises the function f(x,y).
    """
    left_val = fun(left, cparam)
    right_val = fun(right, cparam)
    if left_val > right_val:
        max_val = left_val
        search_max = left
    else:
        max_val = right_val
        search_max = right
    mid = search_max
    mid_val = max_val
    it_count: int = 0
    while abs(left - right) > tol and it_count < max_iter:
        mid = (left + right) / 2
        mid_val = fun(mid, cparam)
        it_count = it_count + 1
        if (mid_val - right_val) < (mid_val - left_val):
            left = mid
            left_val = mid_val
        else:
            right = mid
            right_val = mid_val
    if max_val > mid_val:
        mid = search_max
    return mid


def calc_phase_compensation(touch_node: complex) -> float:
    """
    Calculate the optimum phase compensation for a complex touch delta which maximizes the real.
    This algorithm uses a binary search iterative method.
    :param touch_node: single node touch delta (complex) to which the phase is to be compensated
    :return: phase compensation value
    """
    # define touch magnitude function
    touch_mag_fun = lambda x, y: (complex(math.cos(x), math.sin(x)) * y).real

    # locate extrema at angles 0, pi/2, pi, 3pi/2
    touch_mag = np.zeros(4)
    for ang_idx in range(4):
        ang = ang_idx / 2 * math.pi
        touch_mag[ang_idx] = touch_mag_fun(ang, touch_node)

    # find phase with maximum real magnitude and determine search phase range
    idx = np.argmax(touch_mag)
    start_idx = idx - 1
    end_idx = idx + 1
    ang_start = start_idx / 2 * math.pi
    ang_end = end_idx / 2 * math.pi

    # use binary search to find phase with maximum real magnitude
    ang = fmax_search(touch_mag_fun, touch_node, ang_start, ang_end, 2 * math.pi / 1000, 10)

    # ensure result is in range 0...2pi
    if ang < 0:
        ang = ang + 2 * math.pi
    return ang


def apply_phase_compensation(touch_delta: complex, ang: float) -> float:
    """

    """
    compensated_touch_delta = np.real(complex(np.cos(ang), np.sin(ang)) * touch_delta)
    return compensated_touch_delta


def calc_mut_phase_compensation(touch_sig: np.array, ref_sig: np.array) -> tuple[float, float, float]:
    """
    Calculates the phase angle which maximizes the real mutual touch delta using a complex touch delta measurement.
    It is assumed that there is a single touch.
    :param touch_sig: np.array which holds the mutual touch complex signal.
    :param ref_sig: np.array which holds the mutual reference complex signal.
    :return: float, mutual phase compensation value
    """
    # Find touch delta node with largest magnitude
    size = np.shape(touch_sig)
    num_tx = size[1]
    num_rx = size[0]
    max_touch_delta_abs = 0
    touch_delta_max = 0
    for tx_node in range(num_tx):
        for rx_node in range(num_rx):
            touch_delta = ref_sig[rx_node][tx_node] - touch_sig[rx_node][tx_node]
            touch_delta_abs = abs(touch_delta)
            if touch_delta_abs > max_touch_delta_abs:
                max_touch_delta_abs = touch_delta_abs
                touch_delta_max = touch_delta

    # find phase which maximises the touch delta real magnitude
    ang = calc_phase_compensation(touch_delta_max)
    #compensated_touch_delta = np.real(complex(np.cos(ang), np.sin(ang)) * touch_delta_max)
    compensated_touch_delta = apply_phase_compensation(touch_delta_max, ang)

    return ang, compensated_touch_delta, touch_delta_max


def calc_sct_phase_compensation(touch_sig: np.array, ref_sig: np.array) -> float:
    """
    Calculates the phase angle which maximizes the real selfcap touch delta using a complex touch delta measurement.
    It is assumed that there is a single touch.
    :param touch_sig: np.array which holds the selfcap touch complex signal.
    :param ref_sig: np.array which holds the selfcap reference complex signal.
    :return: float, selfcap phase compensation value
    """
    # Find touch delta node with largest magnitude
    touch_delta = ref_sig - touch_sig
    max_touch_delta_index = np.argmax(abs(touch_delta))
    if touch_delta.size == 1:
        touch_delta_max = touch_delta
    else:
        touch_delta_max = touch_delta[max_touch_delta_index]

    # find phase which maximises the touch delta real magnitude
    ang = calc_phase_compensation(touch_delta_max)
    return ang