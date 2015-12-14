#!/usr/bin/env python3


__author__ = 'André Hollstein'
__version__ = "0.1/20151214"

import glymur
from glob import glob
import numpy as np
from xml.etree.ElementTree import QName
import xml.etree.ElementTree
from scipy.ndimage.interpolation import zoom
from time import time, sleep
from os import path
import os
import errno
import dill
from os import devnull
import sys
import warnings
import argparse
import multiprocessing
import traceback
import gdal
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import builtins
from copy import deepcopy
from psutil import virtual_memory
from tempfile import TemporaryFile
import resource
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity, adjust_gamma
from threading import Thread

import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

# import pyximport
# pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
# from c_digitize import c_digitize


numpy_types = {16: np.float16, 32: np.float32, 64: np.float64}

texts = {
    "welcome":
        """classical Bayesian for Sentinal-2: cB4S2 Version: %s""" % __version__,

    "S2_MSI_granule_path":
        """Path or pattern to S2 granule folders which shall be processed. If not set otherwise by --output_directory, the output masks are written to the ganule folders. This option is mutually incompatible with --tasks_input_file.""",

    "tasks_input_file":
        """Path of input text file with one granule path per line. If not set otherwise by --output_directory, the output masks are written to the ganule folders. This option is mutually incompatible with --tasks_input_file.""",

    "EeooFF":
        """Done with processing, it was a pleasure serving you. """,

    "TT_button_granues_file":
        """This will open a file selection dialog which allows to pick a single file. The selected file should contain only lines with a valid path on each line. Each path should point to a granule in a Sentinel-2 product. A possible example would be:

/[..]/S2A_[..].SAFE/GRANULE/S2A_[..]_T33VVC_N01.05
/[..]/S2A_[..].SAFE/GRANULE/S2A_[..]_T32VPH_N01.05
/[..]/S2A_[..].SAFE/GRANULE/S2A_[..]_T33UUB_N01.05

In this example, [..] denotes an arbitrary, but valid path on your file system.""",

    "TT_button_granule_path":
        """This will open a folder selection dialog where a valid granule path for processing should be selecded, A valid path could be:

/[..]/S2A_[..].SAFE/GRANULE/S2A_[..]_T32VPH_N01.05

where [..] denotes an arbitrary, but valid path on your file system.""",

    "TT_button_classifier_file":
        """TBD""",

    "TT_button_output_folder":
        """TBD""",

    "TT_target_resolution":
        """TBD""",

    "TT_interpolation_order":
        """TBD""",

    "TT_number_of_threads":
        """TBD""",

    "TT_export_to_RGB":
        """Write jpeg image with RGB of Sentinel-2 image.
""",

    "TT_export_to_RGB_mask":
        """Write gray scalse image of the scene and blend with RGB view of the mask. """,

    "TT_number_of_tiles":
        """www""",

    "TT_export_confidence":
        """www""",

    "TT_export_to_RGB_blend":
        """wwww"""
}


def mkdir_p(path_inp):
    try:
        os.makedirs(path_inp)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path_inp):
            pass
        else:
            raise


def __zeros__(shape, dtype, max_mem_frac=0.3):
    def in_memory_array(shape, dtype):
        return np.zeros(shape, dtype)

    def out_memory_array(shape, dtype):
        print("Not enough memory to keep full image -> fall back to memorymap.")
        dat = np.memmap(filename=TemporaryFile(mode="w+b"), dtype=dtype, shape=tuple(shape))
        dat[:] = 0.0
        return dat

    to_gb = 1.0 / 1024.0 ** 3
    mem = virtual_memory().total * to_gb
    arr = np.int(np.prod(np.array(shape, dtype=np.int64)) * np.zeros(1, dtype=dtype).nbytes * to_gb)

    if arr < max_mem_frac * mem:
        try:
            return in_memory_array(shape, dtype)
        except MemoryError as err:
            return out_memory_array(shape, dtype)
    else:
        print("Try to create array of size %.2fGB on a box with %.2fGB memory -> fall back to memorymap." % (arr, mem))
        return out_memory_array(shape, dtype)


class S2_MSI_Image(object):
    def __init__(self, S2_MSI_granule_path,
                 import_bands=('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11',
                               'B12',),
                 namespace="https://psd-12.sentinel2.eo.esa.int/PSD/S2_PDI_Level-1C_Tile_Metadata.xsd",
                 target_resolution=20.0, numpy_type=np.float16,
                 interpolation_order=1, data_mode="dense", driver="Jasper"):
        """
        Reads Sentinel-2 MSI data into numpy array. Images for different channels are resampled to a common sampling
        :param S2_MSI_granule_path: path to granule folder, folder should contain IMG_DATA folder and S2A_[..].xml file
        :param import_bands: list of bands to import, default: ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
                                                                                    'B8A', 'B09', 'B10', 'B11', 'B12',]
        :param namespace: for XML file
        :param target_resolution: spatial resolution in meter
        :param numpy_type: type for result
        :param interpolation_order: integer for interpolation 1,2,3
        :param data_mode: either "dense" or "sparse"
        :param driver: should be "OpenJpeg200","Jasper"
        """

        self.driver = driver
        self.target_resolution = target_resolution
        self.full_band_list = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11',
                               'B12', ]
        self.band_list = list(import_bands)

        print("Load S2 MSI data from:%s" % S2_MSI_granule_path)

        # take XML file in folder above IMG_DATA
        xml_file_name = glob(path.join(S2_MSI_granule_path, "S2*.xml"))[0]

        bf = xml.etree.ElementTree.parse(xml_file_name).getroot()

        # extract geo coding info from XML file
        geo_codings = bf.find(str(QName(namespace, 'Geometric_Info'))).find("Tile_Geocoding")

        self.metadata = {}
        self.metadata["HORIZONTAL_CS_NAME"] = geo_codings.find("HORIZONTAL_CS_NAME").text
        self.metadata["HORIZONTAL_CS_CODE"] = geo_codings.find("HORIZONTAL_CS_CODE").text

        # take XML file in folder above GRANULE Folder
        xml_file_name = glob(path.join(path.dirname(path.dirname(S2_MSI_granule_path)), "S2*.xml"))[0]
        bf = xml.etree.ElementTree.parse(xml_file_name).getroot()

        band_map = {'B1': 'B01', 'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06', 'B7': 'B07',
                    'B8': 'B08',
                    'B9': 'B09', 'B10': 'B10', 'B11': 'B11', 'B12': 'B12', 'B8A': 'B8A'}

        band_list = [band_map[ele.text] for ele in bf.find(".//Band_List").findall("BAND_NAME")]
        quantification_value = int(bf.find(".//QUANTIFICATION_VALUE").text)
        solar_irradiance = [ele.text for ele in bf.find(".//Solar_Irradiance_List").findall("SOLAR_IRRADIANCE")]
        physical_gains = [float(ele.text) for ele in bf.findall(".//PHYSICAL_GAINS")]

        self.scaling_factor = quantification_value
        self.to_radiance = {band: gain for band, gain in zip(band_list, solar_irradiance)}

        # get spatial resolution and image sizes
        self.spatial_samplings = {}
        for size in geo_codings.findall("Size"):
            self.spatial_samplings[float(size.get("resolution"))] = {
                key: int(size.find(key).text) for key in ["NROWS", "NCOLS"]}

        # for each type of resolution, get geopositions
        for geo in geo_codings.findall("Geoposition"):
            self.spatial_samplings[float(geo.get("resolution"))].update(
                {key: int(geo.find(key).text) for key in ["ULX", "ULY", "XDIM", "YDIM"]})
        # build inverse dictionary [shape of image]:spatial sampling
        self.shape_to_resolution = {(values["NCOLS"], values["NROWS"]): spatial_sampling for spatial_sampling, values in
                                    self.spatial_samplings.items()}

        # get file names for each band
        self.band_fns = {self._band_name(fn): fn for fn in
                         glob(path.join(S2_MSI_granule_path, "IMG_DATA", "S2A_*.jp2"))}

        self.final_shape = [self.spatial_samplings[target_resolution][ii] for ii in ("NROWS", "NCOLS")]
        print("Final shape for each channel: %s" % str(self.final_shape))

        if data_mode == "dense":
            self.data = __zeros__(shape=self.final_shape + [len(self.band_list)], dtype=numpy_type)
        elif data_mode == "sparse":
            self.data = __zeros__(shape=list(self.final_shape) + [len(self.full_band_list)], dtype=numpy_type)
        else:
            raise ValueError("data_mode=%s not implemented" % data_mode)

        for iband, band in enumerate(self.band_list):
            t0 = time()
            jpfl = self.__read_img(self.band_fns[band])
            img = np.array(jpfl[:, :], dtype=np.float32) / self.scaling_factor  # / self.to_reflectance[band]
            t1 = time()
            zoom_fac = self.shape_to_resolution[jpfl.shape] / target_resolution

            if data_mode == "dense":
                ii = iband
            elif data_mode == "sparse":
                ii = self.full_band_list.index(band)
            else:
                raise ValueError("data_mode=%s not implemented" % data_mode)

            self.data[:, :, ii] = zoom(input=img, zoom=zoom_fac, order=interpolation_order)
            t2 = time()

            print(
                "Read band %s in %.2fs, pure load time:%.2fs, resample time: %.2fs, zoom: %.3f, final shape: %s, index: %i" %
                (band, t2 - t0, t1 - t0, t2 - t1, zoom_fac, str(self.final_shape), ii))

    @staticmethod
    def _band_name(fn):
        return fn.split(".jp2")[0].split("_")[-1]

    def S2_image_to_rgb(self, rgb_bands=("B11", "B08", "B03"), rgb_gamma=(1.0, 1.0, 1.0), pixel_dd_hist=20, pixel_dd=1,
                        hist_chop_off=0.01):
        S2_rgb = np.zeros(list(self.data[::pixel_dd, ::pixel_dd, 0].shape) + [len(rgb_bands)],
                          dtype=np.float16)

        for i_rgb, (band, gamma) in enumerate(zip(rgb_bands, rgb_gamma)):
            i_band = self.band_list.index(band)
            hh, xx = np.histogram(self.data[::pixel_dd_hist, ::pixel_dd_hist, i_band], bins=100)
            bb = 0.5 * (xx[1:] + xx[:-1])
            hh = hh / np.max(hh)
            lim = (lambda x: (np.min(x), np.max(x)))(bb[hh > hist_chop_off])
            print("Rescale band for RGB image: %i,%s,(%.2f,%.2f)->(0,1)" % (i_rgb, band, lim[0], lim[1]))
            S2_rgb[:, :, i_rgb] = rescale_intensity(image=self.data[::pixel_dd, ::pixel_dd, i_band], in_range=lim,
                                                    out_range=(0.0, 1.0))
            if gamma != 0.0:
                S2_rgb[:, :, i_rgb] = adjust_gamma(np.array(S2_rgb[:, :, i_rgb], dtype=np.float32),
                                                   gamma)

        return S2_rgb

    def __read_img(self, fn):
        print("Reading: %s" % fn)
        if self.driver == "OpenJpeg200":
            img = glymur.Jp2k(fn)
        elif self.driver == "Jasper":
            obj = gdal.Open(fn)
            img = obj.GetRasterBand(1).ReadAsArray()
        else:
            raise ValueError("Driver not supported: %s" % self.driver)

        return img

    @staticmethod
    def save_rgb_image(rgb_img, fn, dpi=100.0):
        fig = plt.figure(figsize=np.array(rgb_img.shape[:2]) / dpi)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax = plt.subplot()
        ax.imshow(rgb_img, interpolation="none")
        ax.set_axis_off()
        plt.savefig(fn, dpi=dpi)
        fig.clear()
        plt.close(fig)


class S2_MSI_Mask(object):
    def __init__(self, S2_img, mask_array, mask_legend, mask_confidence_array=None):
        """
        Sentinel-2 MSI masking object

        :param S2_img:instance of S2_MSI_Image
        :param mask_array: masking result, numpy array
        :param mask_legend: dictionary of mask_id:mask_name
        """
        self.metadata = deepcopy(S2_img.metadata)
        self.geo_coding = deepcopy(S2_img.spatial_samplings[S2_img.target_resolution])

        self.mask_array = mask_array
        self.mask_confidence_array = mask_confidence_array
        self.mask_legend = {key: value for key, value in mask_legend.items()}
        self.mask_legend_inv = {value: key for key, value in mask_legend.items()}

    def mask_rgb_array(self, clf_to_col):
        mask_rgb = np.zeros(list(self.mask_array.shape) + [3], dtype=np.float16)
        for key, col in clf_to_col.items():
            mask_rgb[self.mask_array == key, :] = clf_to_col[key]
        return mask_rgb

    def export_mask_rgb(self, fn_img, clf_to_col, rgb_img):
        mask_rgb = self.mask_rgb_array(clf_to_col)

        dpi = 100.0
        fig = plt.figure(figsize=np.array(rgb_img.shape[:2]) / dpi)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax = plt.subplot()
        ax.imshow(mask_rgb, interpolation="none")
        ax.set_axis_off()
        plt.savefig(fn_img, dpi=dpi)
        fig.clear()
        plt.close(fig)

    def export_mask_blend(self, fn_img, clf_to_col, rgb_img):
        mask_rgb = self.mask_rgb_array(clf_to_col)

        dpi = 100.0
        fig = plt.figure(figsize=np.array(rgb_img.shape[:2]) / dpi)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        ax = plt.subplot()
        # RGB image of scene as background
        ax.imshow(rgb_img, interpolation="none")
        # mask colors above, but transparent
        ax.imshow(mask_rgb, interpolation="none", alpha=0.6)
        ax.set_axis_off()
        plt.savefig(fn_img, dpi=dpi)
        fig.clear()
        plt.close(fig)

    def export_confidence_to_jpeg2000(self, fn_img):
        if self.mask_confidence_array is not None:
            self.mask_confidence_array -= np.min(self.mask_confidence_array)
            self.mask_confidence_array /= np.max(self.mask_confidence_array)
            self.mask_confidence_array *= 100
            _ = glymur.Jp2k(fn_img, data=np.array(self.mask_confidence_array, dtype=np.uint8))

    def export_to_jpeg200(self, fn_img, fn_metadata=None, delimiter=","):
        if fn_img is not None:
            _ = glymur.Jp2k(fn_img, data=np.array(self.mask_array, dtype=np.uint8))
            with open(fn_img + ".pkl", "wb") as fl:
                dill.dump(self.mask_array, fl)

        if fn_metadata is not None:
            with open(fn_metadata, 'w') as outfile:
                for key, value in sorted(self.metadata.items()):
                    outfile.write(str(key) + delimiter + str(value) + '\n')
                for key, value in sorted(self.geo_coding.items()):
                    outfile.write(str(key) + delimiter + str(value) + '\n')


def to_clf(inp):
    """ helper function which sets the type of features and assures numeric values
    :param inp:
    :return: np.float array without NaN's or INF's
    """
    return np.nan_to_num(np.array(inp, dtype=np.float))


def save_divide(d1, d2, mx=100.0):
    """ save division without introducing NaN's
    :param d1:
    :param d2:
    :return: d1/d2
    """
    dd1 = to_clf(d1)
    dd2 = to_clf(d2)
    dd2[dd2 == 0.0] = 1e-6
    dd1 /= dd2
    dd1 = np.nan_to_num(dd1)
    dd1[dd1 > mx] = mx
    dd1[dd1 < -mx] = -mx
    return dd1


"""
this is just an example on how one could define classification
functions, this is an argument to the ToClassifier Classes
"""
clf_functions = {
    "ratio": lambda d1, d2: save_divide(d1, d2),
    "index": lambda d1, d2: save_divide(d1 - d2, d1 + d2),
    "difference": lambda d1, d2: to_clf(d1) - to_clf(d2),
    "channel": lambda d: to_clf(d),
    "depth": lambda d1, d2, d3: save_divide(to_clf(d1) + to_clf(d2), d3),
    "index_free_diff": lambda d1, d2, d3, d4: save_divide(to_clf(d1) - to_clf(d2), to_clf(d3) - to_clf(d4)),
    "index_free_add": lambda d1, d2, d3, d4: save_divide(to_clf(d1) + to_clf(d2), to_clf(d3) + to_clf(d4)),
}


class _ToClassifierBase(object):
    """
    internal base class for generation of classifiers, only to use common __call__
    """

    def __init__(self):
        """ dummy __init__ which sets all basic needed attributes to none,
        need derived classes to implement proper __init__
        :return:
        """
        self.n_classifiers = None
        self.classifiers_fk = None
        self.classifiers_id = None
        self.clf_functions = None
        self.classifiers_id_full = None

    def adjust_classifier_ids(self, full_bands, band_lists):
        self.classifiers_id = [np.array([band_lists.index(full_bands[ii]) for ii in clf], dtype=np.int)
                               for clf in self.classifiers_id_full]
        print("""Adjusting classifier channel list indices to actual image, convert from:
%s to \n %s. \n This results in a changed classifier index array from:""" % (str(full_bands), str(band_lists)))
        for func, old, new in zip(self.classifiers_fk, self.classifiers_id_full, self.classifiers_id):
            print("%s : %s -> %s" % (func, old, new))

    @staticmethod
    def list_np(arr):
        """
        This is fixing a numpy annoyance where scalar arrays and vectors arrays are treated differently,
        namely one can not iterate over a scalar, this function fixes this in that a python list is
        returned for both scalar and vector arrays
        :param arr: numpy array or numpy scalar
        :return: list with values of arr
        """
        try:
            return list(arr)
        except TypeError:  # will fail if arr is numpy scalar array
            return list(arr.reshape(1))

    def __call__(self, data):
        """
        Secret sauce of the Classical Bayesian approach in python, here the input data->([n_samples,n_data_channels])
        are transformed into ret->([n_samples,n_classifiers])
        Iteration is performed over classifiers_fk (name found in clf_functions) and classifiers_id
        (channel selection from data for this function)
        :param data: n_samples x n_data_channels
        :return: res: n_samples x n_classifiers
        """

        # ret = np.zeros((data.shape[0], self.n_classifiers))  # initialize result
        ret = __zeros__(shape=(data.shape[0], self.n_classifiers), dtype=np.float32)  # initialize result
        for ii, (fn, idx_clf) in enumerate(zip(self.classifiers_fk, self.classifiers_id)):
            # note that that input of clf_function[fn] is a generator expression where
            # iteration is performed over the selected classifiers_id's
            ret[:, ii] = self.clf_functions[fn](*(data[:, ii] for ii in self.list_np(idx_clf)))
        return ret


class ToClassifierDef(_ToClassifierBase):
    """
    Most simple case of a usable ToClassifier instance, everything is fixed
    """

    def __init__(self, classifiers_id, classifiers_fk, clf_functions):
        """
        classifiers_id: list of lists/np.arrays with indices which are inputs for classifier functions
        classifiers_fk: list of names for the functions to be used
        clf_functions: dictionary for key, value pairs of function names as used in classifiers_fk
        """
        self.n_classifiers = len(classifiers_fk)
        self.clf_functions = clf_functions
        self.classifiers_id_full = classifiers_id
        self.classifiers_id = classifiers_id
        self.classifiers_fk = classifiers_fk

        # assert equal length
        assert len(self.classifiers_id) == self.n_classifiers
        assert len(self.classifiers_fk) == self.n_classifiers
        # assert that used functions are in self.clf_functions
        for cl_fk in self.classifiers_fk:
            assert cl_fk in self.clf_functions
        # assert that each value in the dict is a callable
        for name, func in self.clf_functions.items():
            if hasattr(func, "__call__") is False:
                raise ValueError("Each value in clf_functions should be a callable, error for: %s" % name)


class ClassicalBayesian(object):
    def __init__(self, mk_clf, bns, hh_full, hh, hh_n, n_bins, classes, n_classes, bb_full):

        self.mk_clf = mk_clf
        self.bns = bns
        self.hh_full = hh_full
        self.hh = hh
        self.hh_n = hh_n
        self.n_bins = n_bins
        self.classes = classes
        self.n_classes = n_classes
        self.bb_full = bb_full

    def __in_bounds__(self, ids):
        ids[ids > self.n_bins - 1] = self.n_bins - 1

    def __predict__(self, xx):
        ids = [np.digitize(ff, bb) - 1 for ff, bb in zip(self.mk_clf(xx).transpose(), self.bb_full)]
        # ids = [c_digitize(ff, bb) - 1 for ff, bb in zip(self.mk_clf(xx).transpose(), self.bb_full)]

        for ii in ids:
            self.__in_bounds__(ii)
        pp = np.zeros((self.n_classes, len(ids[0])), dtype=np.float)
        for ii, cl in enumerate(self.classes):
            hh = self.hh[cl][ids]
            hh_full = self.hh_full[ids]
            hh_valid = hh_full > 0.0
            pp[ii, hh_valid] = hh[hh_valid] / hh_full[hh_valid] / self.n_classes
        return pp

    def predict_proba(self, xx):
        pr = self.__predict__(xx.reshape((-1, xx.shape[-1]))).transpose()
        return pr.reshape(list(xx.shape[:-1]) + [pr.shape[-1], ])

    def predict(self, xx):
        pr = self.classes[np.argmax(self.__predict__(xx.reshape((-1, xx.shape[-1]))), axis=0)]
        return pr.reshape(xx.shape[:-1])

    def conf(self, xx):
        proba = self.predict_proba(xx)
        conf = np.nan_to_num(np.max(proba, axis=1) / np.sum(proba, axis=1))
        return conf.reshape(xx.shape[:-1])

    def predict_and_conf(self, xx):
        proba = self.__predict__(xx.reshape((-1, xx.shape[-1]))).transpose()
        conf = np.nan_to_num(np.max(proba, axis=1) / np.sum(proba, axis=1))
        pr = self.classes[np.argmax(proba, axis=1)]
        return pr.reshape(xx.shape[:-1]), conf.reshape(xx.shape[:-1])


class S2_MSI_Classifier(object):
    def __init__(self, cb_clf, mask_legend, clf_to_col, processing_tiles=10):
        self.cb_clf = cb_clf
        self.mask_legend = mask_legend
        self.clf_to_col = clf_to_col
        self.processing_tiles = processing_tiles

        self.S2_MSI_channels = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11',
                                'B12']

        unique_channel_list = []
        for clf_ids in cb_clf.mk_clf.classifiers_id:
            unique_channel_list += list(clf_ids)
        self.unique_channel_ids = np.unique(unique_channel_list)
        self.unique_channel_str = [self.S2_MSI_channels[ii] for ii in self.unique_channel_ids]

    def __call__(self, S2_img):
        self.cb_clf.mk_clf.adjust_classifier_ids(full_bands=S2_img.full_band_list,
                                                 band_lists=S2_img.band_list)
        if self.processing_tiles == 0:
            # mask_array = self.cb_clf.predict(S2_img.data)
            mask_array, mask_conf = self.cb_clf.predict_and_conf(S2_img.data)
        else:
            mask_array = np.zeros(S2_img.data.shape[:2], dtype=np.int)
            mask_conf = np.zeros(S2_img.data.shape[:2], dtype=np.float16)

            line_segs = np.linspace(0, S2_img.data.shape[0], self.processing_tiles, dtype=np.int)
            for ii, (i1, i2) in enumerate(zip(line_segs[:-1], line_segs[1:])):
                print("Processing lines segment %i of %i -> %i:%i" % (ii + 1, self.processing_tiles, i1, i2))
                # mask_array[i1:i2,:] = self.cb_clf.predict(S2_img.data[i1:i2,:,:])
                mask_array[i1:i2, :], mask_conf[i1:i2, :] = self.cb_clf.predict_and_conf(S2_img.data[i1:i2, :, :])

        mask_array[S2_img.data[:, :, 0] == 0.0] = 0
        mask_conf[S2_img.data[:, :, 0] == 0.0] = 0

        return S2_MSI_Mask(S2_img=S2_img, mask_array=mask_array,
                           mask_legend=self.mask_legend, mask_confidence_array=mask_conf)


def mask_image(args, S2_MSI_granule_path):
    global S2_clf

    def fnly():
        try:
            return sys.stdout.lines, sys.stderr.lines
        except:
            return None

    if args.output_directory == "":
        path_output = path.join(S2_MSI_granule_path, "IMG_DATA")
    else:
        path_output = args.output_directory

    if args.create_output_folder is True:
        mkdir_p(path_output)

    basename_output = path.basename(S2_MSI_granule_path)[:-7]

    outs = glob(path.join(path_output, basename_output + "*"))
    if len(outs) > 0:
        print("Some Output already exists:")
        for out in outs:
            print(out)

        if args.overwrite_output is True:
            print("Continue overwriting existing files!")
        else:
            print("Stop here since output already exists.")
            return fnly()

    import_bands = set(S2_clf.unique_channel_str)
    if args.export_RGB is True:
        import_bands = set(list(import_bands) + args.RGB_channels.split(","))

    t0 = time()
    S2_img = S2_MSI_Image(S2_MSI_granule_path=S2_MSI_granule_path,
                          import_bands=import_bands,
                          data_mode="dense",
                          target_resolution=args.target_resolution,
                          numpy_type=numpy_types[args.float_type],
                          interpolation_order=args.interpolation_order
                          )
    print("Total time to read S2 image data: %.2fs" % (time() - t0,))

    print("Start detection.")
    t0 = time()
    S2_msk = S2_clf(S2_img)
    t1 = time()
    print("Detection performed in %.2fs -> %.2f Mpx/s" % (
        t1 - t0, S2_img.data.shape[0] * S2_img.data.shape[1] / 10 ** 6 / (t1 - t0)))

    if args.export_RGB is True:
        rgb_img = S2_img.S2_image_to_rgb(rgb_bands=args.RGB_channels.split(","),
                                         pixel_dd=args.additional_output_pixel_skip)
        fn = path.join(path_output, "%s_RGB.jpg" % basename_output)
        print("Write RGB Image to: %s" % fn)
        S2_img.save_rgb_image(rgb_img=rgb_img, fn=fn)

    if args.export_confidence is True:
        fn_img = path.join(path_output, "%s_CONF.jp2" % basename_output)
        print("Write output to: %s" % fn_img)
        S2_msk.export_confidence_to_jpeg2000(fn_img=fn_img)

    if args.export_mask_blend is True:
        rgb_img = S2_img.S2_image_to_rgb(rgb_bands=args.RGB_channels.split(","))
        fn = path.join(path_output, "%s_MASK_BLEND.jpg" % basename_output)
        print("Write MASK with blended gray scale Image to: %s" % fn)
        S2_msk.export_mask_blend(fn, clf_to_col=S2_clf.clf_to_col, rgb_img=rgb_img)
        del rgb_img

    if args.export_mask_rgb is True:
        rgb_img = S2_img.S2_image_to_rgb(rgb_bands=args.RGB_channels.split(","))
        fn = path.join(path_output, "%s_MASK_RGB.jpg" % basename_output)
        print("Write MASK RGB Image to: %s" % fn)
        S2_msk.export_mask_rgb(fn, clf_to_col=S2_clf.clf_to_col, rgb_img=rgb_img)
        del rgb_img

    if args.mask_export_format == "jp2":
        fn_img = path.join(path_output, "%s_MASK.jp2" % basename_output)
        fn_metadata = path.join(path_output, "%s.csv" % basename_output)
        print("Write output to: %s" % fn_img)
        print("Write output to: %s" % fn_metadata)
        S2_msk.export_to_jpeg200(fn_img=fn_img, fn_metadata=fn_metadata)
    else:
        raise ValueError("mask_export_format=%s is not understood" % args.mask_export_format)

    return fnly()


def main(args):
    global Pool
    global ThreadPool

    if args.verbosity == 0:
        ff = open(devnull, 'w')
        sys.stdout = ff

    print(texts["welcome"])

    if args.show_warnings is not True:
        print("Switch off printing of warnings from python packages")
        warnings.filterwarnings("ignore")

    print("Read classical Bayesian persistence file: %s" % args.persistence_file)
    with open("./cb_data.pkl", "rb") as fl:
        mask_legend = dill.load(fl)
        clf_to_col = dill.load(fl)
        data_cB, data_mk_clf = dill.load(fl)
    data_mk_clf["clf_functions"] = clf_functions

    data_cB.update({"mk_clf": ToClassifierDef(**data_mk_clf)})
    cb_clf = ClassicalBayesian(**data_cB)

    builtins.S2_clf = S2_MSI_Classifier(cb_clf=cb_clf, mask_legend=mask_legend,
                               processing_tiles=args.processing_tiles,
                               clf_to_col=clf_to_col)

    if args.S2_MSI_granule_path is None:
        args.S2_MSI_granule_path = glob(args.glob_search_pattern, recursive=True)
        print("No Input data given -> traverse local path and search for granules:")
        for granule in args.S2_MSI_granule_path:
            print(granule)

    tasks = [(args, granule) for granule in args.S2_MSI_granule_path]

    if args.number_of_threads == 1:
        print("Start processing of %i jobs." % len(args.S2_MSI_granule_path))
        for granule in args.S2_MSI_granule_path:
            mask_image(args, granule)
    elif args.number_of_threads > 1:
        print(
            "Start processing of %i jobs using %i processes." % (len(args.S2_MSI_granule_path), args.number_of_threads))

        def _init_():
            globals()["update"] = lambda: None  # tkinter is not able to digest multiple threads
            sys.stdout = StdoutToList()
            sys.stderr = StdoutToList()

        print("Start Processing in parallel running threads. Output of individual jobs is shown when done. ")

        if args.use_thread_pool is True:
            print("!!!!!!!!!!!! Use ThreadPool  !!!!!!!!!")
            Pool = ThreadPool

        pool = Pool(initializer=_init_, processes=args.number_of_threads)
        jobs = [pool.apply_async(mask_image, task) for task in tasks]

        while len(jobs) > 0:
            sleep(5.0)
            readies = [job.ready() for job in jobs]
            for job, ready in zip(jobs, readies):
                if ready is True:
                    lne_out, lne_err = job.get()
                    print("### print jop output ###")
                    for line in lne_out:
                        print(line)
                    if len(lne_err) > 1:
                        print("### errors occurred -> print stderr: ###")
                        for line in lne_out:
                            print(line)
            jobs[:] = [job for job, ready in zip(jobs, readies) if ready is False]
            print("#> Open Jobs: %i <#" % len(jobs))
        pool.close()

    else:
        raise ValueError("The number of threads should be larger or equal to zero and not:%i" % args.number_of_threads)

    print(texts["EeooFF"])


class StdRedirector(object):
    def __init__(self, widget, gui=None):
        self.widget = widget
        self.counter = 0
        self.gui = gui

    def write(self, string):
        # self.widget.insert(tk.END,string)
        self.counter += 1
        # self.widget.insert(tk.END, "%s:%s" % (str(ctime()), str(string).strip()))
        # self.widget.insert(tk.END, string)

        rr = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024. ** 2
        ss = str(string).rstrip()

        if len(ss) > 0:
            self.widget.insert(tk.END, "mem: %.2f GB : %s\n" % (rr, ss))
            self.widget.see(tk.END)

        if self.gui.update is not None:
            self.gui.update()

    def flush(self):
        pass


class StdoutToList(object):
    def __init__(self, ):
        self.lines = []

    def write(self, string):
        self.lines.append(string)

    def flush(self):
        pass


class ToolTip:
    def __init__(self, master, text='Your text here', delay=150, **opts):
        self.master = master
        self._opts = {
            'anchor': 'center',
            'bd': 1,
            'bg': 'lightyellow',
            'delay': delay,
            'fg': 'black',
            'follow_mouse': 0,
            'font': None,
            'justify': 'left',
            'padx': 4,
            'pady': 2,
            'relief': 'solid',
            'state': 'normal',
            'text': text,
            'textvariable': None,
            'width': 0,
            'wraplength': 500
        }
        self.configure(**opts)
        self._tipwindow = None
        self._id = None
        self._id1 = self.master.bind("<Enter>", self.enter, '+')
        self._id2 = self.master.bind("<Leave>", self.leave, '+')
        self._id3 = self.master.bind("<ButtonPress>", self.leave, '+')
        self._follow_mouse = 0
        if self._opts['follow_mouse']:
            self._id4 = self.master.bind("<Motion>", self.motion, '+')
            self._follow_mouse = 1

    def configure(self, **opts):
        for key in opts:
            if key in self._opts:
                self._opts[key] = opts[key]
            else:
                keyerror = 'KeyError: Unknown option: "%s"' % key
                raise keyerror

    def enter(self, event=None):  # handles <Enter> event
        self._schedule()

    def leave(self, event=None):  # handles <Leave> event
        self._unschedule()
        self._hide()

    def motion(self, event=None):  # handles <Motion> event
        if self._tipwindow and self._follow_mouse:
            x, y = self.coords()
            self._tipwindow.wm_geometry("+%d+%d" % (x, y))

    def _schedule(self):
        self._unschedule()
        if self._opts['state'] == 'disabled':
            return
        self._id = self.master.after(self._opts['delay'], self._show)

    def _unschedule(self):
        idd = self._id
        self._id = None
        if idd:
            self.master.after_cancel(idd)

    def _show(self):
        if self._opts['state'] == 'disabled':
            self._unschedule()
            return
        if not self._tipwindow:
            self._tipwindow = tw = tk.Toplevel(self.master)
            # hide the window until we know the geometry
            tw.withdraw()
            tw.wm_overrideredirect(1)

            if tw.tk.call("tk", "windowingsystem") == 'aqua':
                tw.tk.call("::tk::unsupported::MacWindowStyle", "style", tw._w, "help", "none")

            self.create_contents()
            tw.update_idletasks()
            x, y = self.coords()
            tw.wm_geometry("+%d+%d" % (x, y))
            tw.deiconify()

    def _hide(self):
        tw = self._tipwindow
        self._tipwindow = None
        if tw:
            tw.destroy()

    def coords(self):
        # The tip window must be completely outside the master widget;
        # otherwise when the mouse enters the tip window we get
        # a leave event and it disappears, and then we get an enter
        # event and it reappears, and so on forever :-(
        # or we take care that the mouse pointer is always outside the tipwindow :-)
        tw = self._tipwindow
        twx, twy = tw.winfo_reqwidth(), tw.winfo_reqheight()
        w, h = tw.winfo_screenwidth(), tw.winfo_screenheight()
        # calculate the y coordinate:
        if self._follow_mouse:
            y = tw.winfo_pointery() + 20
            # make sure the tipwindow is never outside the screen:
            if y + twy > h:
                y = y - twy - 30
        else:
            y = self.master.winfo_rooty() + self.master.winfo_height() + 3
            if y + twy > h:
                y = self.master.winfo_rooty() - twy - 3
        # we can use the same x coord in both cases:
        x = tw.winfo_pointerx() - twx / 2
        if x < 0:
            x = 0
        elif x + twx > w:
            x = w - twx
        return x, y

    def create_contents(self):
        opts = self._opts.copy()
        for opt in ('delay', 'follow_mouse', 'state'):
            del opts[opt]
        label = tk.Label(self._tipwindow, **opts)
        label.pack()


# noinspection PyAttributeOutsideInit
class Gui(tk.Tk):
    def __init__(self, parent, args):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.args = args
        self.init_gui()

    def init_gui(self):

        self.title(texts["welcome"])

        tk.Grid.columnconfigure(self, 0, weight=0)
        tk.Grid.columnconfigure(self, 1, weight=1)

        i_row = 0
        tk_lab = tk.Label(self, text="General setup:")
        tk_lab.grid(row=i_row, sticky=tk.E)

        frame = tk.Frame(self)
        frame.grid(row=i_row, column=1, sticky=tk.E + tk.W)
        for ii in range(0, 4):
            tk.Grid.columnconfigure(frame, ii, weight=1)

        self.tk_button1 = tk.Button(frame, text='get granules file', command=self.get_input_file)
        self.tk_button1.grid(row=1, column=0, sticky=tk.E + tk.W + tk.N + tk.S)
        ToolTip(self.tk_button1, text=texts["TT_button_granues_file"])

        self.tk_button2 = tk.Button(frame, text='get path to GRANULE folder', command=self.get_granule_folder)
        self.tk_button2.grid(row=1, column=1, sticky=tk.E + tk.W + tk.N + tk.S)
        ToolTip(self.tk_button2, text=texts["TT_button_granule_path"])

        self.tk_button3 = tk.Button(frame, text='get classifier file', command=self.get_classifier_file)
        self.tk_button3.grid(row=1, column=2, sticky=tk.E + tk.W + tk.N + tk.S)
        ToolTip(self.tk_button3, text=texts["TT_button_classifier_file"])

        self.tk_button4 = tk.Button(frame, text='select output folder', command=self.get_output_folder)
        self.tk_button4.grid(row=1, column=3, sticky=tk.E + tk.W + tk.N + tk.S)
        ToolTip(self.tk_button4, text=texts["TT_button_output_folder"])

        i_row += 1
        tk.Label(self, text="Search Pattern:").grid(row=i_row, sticky=tk.E, column=0)
        frame = tk.Frame(self)
        frame.grid(row=i_row, column=1, sticky=tk.E + tk.W)
        self.tk_entry_output_folder = tk.Entry(frame)
        self.tk_entry_output_folder.insert(0, args.glob_search_pattern)
        self.tk_entry_output_folder.pack(side=tk.LEFT, fill=tk.X, expand=1)
        self.tk_entry_output_folder.bind('<Return>', self.test_pattern)
        self.tk_button5 = tk.Button(frame, text="Test Pattern", command=self.test_pattern)
        self.tk_button5.pack(side=tk.LEFT)

        i_row += 1
        tk_lab = tk.Label(self, text="Target Resolution:")
        tk_lab.grid(row=i_row, sticky=tk.E)
        frame = tk.Frame(self)
        frame.grid(row=i_row, column=1, sticky=tk.E + tk.W)
        ToolTip(frame, text=texts["TT_target_resolution"])

        self.tr = tk.DoubleVar()
        self.tr.set(20.0)
        tk.Radiobutton(frame, text="10m", variable=self.tr, value=10.0).pack(side=tk.LEFT, fill=tk.X)
        tk.Radiobutton(frame, text="20m", variable=self.tr, value=20.0).pack(side=tk.LEFT, fill=tk.X)
        tk.Radiobutton(frame, text="60m", variable=self.tr, value=60.0).pack(side=tk.LEFT, fill=tk.X)

        separator = tk.Frame(frame, height=2, bd=5, relief=tk.SUNKEN)
        separator.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        tk_lab = tk.Label(frame, text="Interpolation order:")
        tk_lab.pack(side=tk.LEFT, fill=tk.X)
        ToolTip(tk_lab, text=texts["TT_interpolation_order"])
        self.io = tk.IntVar()
        self.io.set(1)
        for ii, jj in enumerate(range(1, 6)):
            tk.Radiobutton(frame, text="%i" % jj, variable=self.io, value=jj).pack(side=tk.LEFT, fill=tk.X)

        i_row += 1
        tk_lab = tk.Label(self, text="Export Options:")
        tk_lab.grid(row=i_row, sticky=tk.E)
        frame = tk.Frame(self)
        frame.grid(row=i_row, column=1, sticky=tk.E + tk.W)
        et = tk.StringVar()
        et.set("jp2")
        for ii, (key, text) in enumerate([("jp2", "Jpeg200+CSV")]):
            tk.Radiobutton(frame, text="%s" % text, variable=et, value=key).pack(side=tk.LEFT, fill=tk.X)

        frame = tk.Frame(frame, padx=1, bd=1)
        frame.pack(side=tk.LEFT, fill=tk.X)

        tk_lab = tk.Label(frame, text="Optional exports:")
        tk_lab.pack(side=tk.LEFT, fill=tk.X)
        self.exp_to_rgb = tk.IntVar()
        self.exp_to_rgb.set(1)
        self.check_rgb = tk.Checkbutton(frame, text="RGB Image", onvalue=1, offvalue=0, variable=self.exp_to_rgb)
        self.check_rgb.pack(side=tk.LEFT, fill=tk.X)
        ToolTip(self.check_rgb, text=texts["TT_export_to_RGB"])

        self.exp_to_rgb_mask = tk.IntVar()
        self.exp_to_rgb_mask.set(1)
        self.check_rgb_mask = tk.Checkbutton(frame, text="MASK-RGB", onvalue=1, offvalue=0,
                                             variable=self.exp_to_rgb_mask)
        self.check_rgb_mask.pack(side=tk.LEFT, fill=tk.X)
        ToolTip(self.check_rgb_mask, text=texts["TT_export_to_RGB_mask"])

        self.exp_to_blend_mask = tk.IntVar()
        self.exp_to_blend_mask.set(1)
        self.check_blend_mask = tk.Checkbutton(frame, text="MASK-BLEND", onvalue=1, offvalue=0,
                                               variable=self.exp_to_blend_mask)
        self.check_blend_mask.pack(side=tk.LEFT, fill=tk.X)
        ToolTip(self.check_blend_mask, text=texts["TT_export_to_RGB_blend"])

        self.exp_confidence = tk.IntVar()
        self.exp_confidence.set(1)
        self.check_confidence = tk.Checkbutton(frame, text="confidence", onvalue=1, offvalue=0,
                                               variable=self.exp_confidence)
        self.check_confidence.pack(side=tk.LEFT, fill=tk.X)
        ToolTip(self.check_confidence, text=texts["TT_export_confidence"])

        i_row += 1
        tk_lab = tk.Label(self, text="Performance:")
        tk_lab.grid(row=i_row, sticky=tk.E)
        frame = tk.Frame(self)
        frame.grid(row=i_row, column=1, sticky=tk.E + tk.W)

        self.oo = tk.IntVar()
        self.oo.set(0)
        tk.Checkbutton(frame, text="Overwrite Output", onvalue=1, offvalue=0, variable=self.oo).pack(side=tk.LEFT,
                                                                                                     fill=tk.X, padx=5)

        tk.Label(frame, text="Threads:").pack(side=tk.LEFT, fill=tk.X)
        self.tk_scale = tk.Scale(frame, from_=1, to=multiprocessing.cpu_count(), resolution=1, orient=tk.HORIZONTAL)
        self.tk_scale.pack(side=tk.LEFT, fill=tk.X, expand=1)
        ToolTip(self.tk_scale, text=texts["TT_number_of_threads"])

        tk.Label(frame, text="Tiles:").pack(side=tk.LEFT, fill=tk.X, padx=5)
        self.tk_scale_tiles = tk.Scale(frame, from_=0, to=20, resolution=1, orient=tk.HORIZONTAL)
        self.tk_scale_tiles.pack(side=tk.LEFT, fill=tk.X, expand=1)
        ToolTip(self.tk_scale_tiles, text=texts["TT_number_of_tiles"])

        i_row += 1
        self.tk_button_run = tk.Button(self, text="Run Program", command=lambda: self.main_gui())
        self.tk_button_run.grid(row=i_row, columnspan=2, sticky=tk.E + tk.W)
        tk.Grid.rowconfigure(self, i_row, weight=0)

        self.buttons_deactivate_while_processing = [self.tk_button1, self.tk_button2, self.tk_button3, self.tk_button4,
                                                    self.tk_button_run, self.tk_button5]

        i_row += 1
        tk.Grid.rowconfigure(self, i_row, weight=1)
        frame = tk.Frame(self)
        frame.grid(row=i_row, column=0, columnspan=2, sticky=tk.E + tk.W + tk.N + tk.S)
        sb = tk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        tk_text = tk.Text(frame, wrap=tk.WORD, background="black", foreground="white",
                          yscrollcommand=sb.set)
        tk_text.pack(fill=tk.BOTH, expand=1)
        sb.config(command=tk_text.yview)

        sys.stdout = StdRedirector(tk_text, gui=self)
        sys.stderr = StdRedirector(tk_text, gui=self)

    def update_gui(self):
        while self.updateNeeded:
            self.update()
            sleep(0.05)

    def main_gui(self):
        self.updateNeeded = True
        self.updateGUIThread = Thread(target=self.update_gui)
        self.updateGUIThread.start()

        for button in self.buttons_deactivate_while_processing:
            button["state"] = "disabled"

        self.args.target_resolution = self.tr.get()
        self.args.number_of_threads = self.tk_scale.get()
        self.args.processing_tiles = self.tk_scale_tiles.get()
        self.args.interpolation_order = self.io.get()
        self.args.export_RGB = True if self.exp_to_rgb.get() == 1 else False
        self.args.export_confidence = True if self.exp_confidence.get() == 1 else False
        self.args.export_mask_blend = True if self.exp_to_rgb_mask.get() == 1 else False
        self.args.export_mask_rgb = True if self.exp_to_blend_mask.get() == 1 else False
        self.args.overwrite_output = True if self.oo.get() == 1 else False

        self.args.glob_search_pattern = self.tk_entry_output_folder.get()

        print("Start with following settings:")
        for key, value in self.args.__dict__.items():
            print("%s -> %s" % (key, str(value)))
            self.update()

        try:
            main(args=self.args)
        except Exception as err:
            print("### Error occurred #####")
            print(err)
            print(repr(err))
            print(traceback.print_tb(err.__traceback__))
            pass

        for button in self.buttons_deactivate_while_processing:
            button["state"] = "normal"

        self.updateNeeded = False

    def test_pattern(self, event=None):
        pat = self.tk_entry_output_folder.get()
        res = glob(pat, recursive=True)
        print("Test search patterns: %s, results: %i" % (pat, len(res)))
        for ii, rr in enumerate(res):
            print("%i,%s" % (ii, rr))

    def get_input_file(self):
        file_name = askopenfilename(multiple=0, title="Pick file with list of input GRANULE directories.")
        lines = open(file_name, "r").readlines()
        print("Path: in: %s" % file_name)
        self.args.S2_MSI_granule_path = []
        for ii, line in enumerate(lines):
            pp = line.strip()
            if path.isdir(pp):
                print("%i:%s" % (ii, pp))
                self.args.S2_MSI_granule_path.append(pp)
            else:
                print("%i -> not a directory, ignore: %s" % (ii, pp))

    def get_granule_folder(self):
        dir_name = askdirectory(
            title='path to GRANULE folder, something like */S2A_OPER_PRD_[..].SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_[...]_N01.03/')
        self.args.S2_MSI_granule_path = [dir_name]
        print("Selected directory: %s" % dir_name)

    def get_classifier_file(self):
        file_name = askopenfilename(multiple=0, title="Pick file to define classifier, should be *.pkl")
        print("New classifier persistence file selected: %s" % file_name)
        self.args.persistence_file = file_name

    def get_output_folder(self):
        dir_name = askdirectory(title='Select output folder.')
        self.args.output_directory = dir_name
        print("Selected directory for output: %s" % dir_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='GFZ-detection',
                                     description='Cloud, Cirrus, Snow, Shadow Detection for Sentinel-2. ')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-i", "--S2_MSI_granule_path", action="store", type=str, nargs="*",
                       help=texts["S2_MSI_granule_path"])
    group.add_argument("-f", "--tasks_input_file", action="store", type=str, help=texts["tasks_input_file"])

    parser.add_argument("-r", "--target_resolution", help="sdsd", action="store", type=float, default=20.0,
                        required=False)
    parser.add_argument("-o", "--interpolation_order", help="sdsd", action="store", type=int, default=1, required=False,
                        choices=range(1, 6))
    parser.add_argument("-p", "--persistence_file", help="asd", action="store", type=str,
                        default="./cb_data_20151211.pkl")  # "./cb_data_20151210.pkl")
    parser.add_argument("-e", "--mask_export_format", help="asd", action="store", type=str, default="jp2",
                        choices=["jp2"])
    parser.add_argument("-w", "--show_warnings", help="asd", action="store_true", default=False)
    parser.add_argument("-t", "--float_type", help="sdsd", action="store", type=int, default=32, required=False,
                        choices=[16, 32, 64])
    parser.add_argument("-d", "--output_directory", help="asd", action="store", type=str, default="./")
    parser.add_argument("-m", "--create_output_folder", help="asd", action="store_false", default=True)
    parser.add_argument("-v", "--verbosity", help="asd", action="store", type=int, choices=[0, 1], default=1)
    parser.add_argument("-n", "--number_of_threads", help="asd", action="store", type=int, default=1)
    parser.add_argument("-q", "--use_thread_pool", help="asd", action="store_true", default=False)
    parser.add_argument("-j", "--RGB_channels", help="asd", action="store", type=str, default="B11,B08,B03")
    parser.add_argument("-g", "--gui", help="asd", action="store", type=str, default="yes", choices=["yes", "no"])

    parser.add_argument("-x", "--export_RGB", help="asd", action="store_true", default=False)
    parser.add_argument("-z", "--export_mask_blend", help="asd", action="store_true", default=False)
    parser.add_argument("-B", "--export_mask_rgb", help="asd", action="store_true", default=False)
    parser.add_argument("-C", "--export_confidence", help="asd", action="store_true", default=False)

    parser.add_argument("-a", "--additional_output_pixel_skip", help="sdsd", action="store", type=int, default=1,
                        required=False)
    parser.add_argument("-T", "--processing_tiles", help="asd", action="store", type=int, choices=range(0, 20),
                        default=10)
    parser.add_argument("-G", "--glob_search_pattern", help="asd", action="store", type=str, default="**/GRANULE/*")

    parser.add_argument("-W", "--overwrite_output", help="asd", action="store_true", default=False)

    args = parser.parse_args()

    if args.S2_MSI_granule_path is not None or args.tasks_input_file is not None and args.gui == "no":
        main(args)
    else:
        print("No command line argument were given -> start GUI")
        gui = Gui(parent=None, args=args)
        gui.mainloop()
