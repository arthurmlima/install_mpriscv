
import ctypes
import numpy as np
import cv2
from flask import Flask, Response


def mpriscv(sel_img):
    my_lib = ctypes.CDLL("../mpriscv/mpriscv.so")
    my_lib.mpriscv.argtypes = [
        ctypes.c_int,                       # sel_img
        ctypes.POINTER(ctypes.c_uint64),    # t0
        ctypes.POINTER(ctypes.c_uint64),    # t1
        ctypes.POINTER(ctypes.c_uint64),    # t2
        ctypes.POINTER(ctypes.c_uint64),    # t3
        ctypes.POINTER(ctypes.c_uint64),    # t4
        ctypes.POINTER(ctypes.c_uint64)     # t5
    ]
    my_lib.mpriscv.restype = ctypes.POINTER(ctypes.c_uint8)
    t0 = ctypes.c_uint64(0)
    t1 = ctypes.c_uint64(0)
    t2 = ctypes.c_uint64(0)
    t3 = ctypes.c_uint64(0)
    t4 = ctypes.c_uint64(0)
    t5 = ctypes.c_uint64(0)
    
    result = my_lib.mpriscv(sel_img, ctypes.byref(t0), ctypes.byref(t1), ctypes.byref(t2), ctypes.byref(t3), ctypes.byref(t4), ctypes.byref(t5))
    array = ctypes.cast(result, ctypes.POINTER(ctypes.c_uint8 * 240 * 240)).contents
    image_data = np.array(array, dtype=np.uint8)
    image_data = np.reshape(image_data, (240, 240))
    return cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB), t0, t1, t2, t3, t4, t5
