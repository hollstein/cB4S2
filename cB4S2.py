#!/usr/bin/env python3

__author__ = 'André Hollstein'
__version__ = "0.1/20151214"

from glob import glob
import numpy as np
from time import time, sleep, gmtime
from os import path
import logging
import sys
import warnings
import argparse
import multiprocessing
import traceback
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import builtins
import matplotlib
from stopit import ThreadingTimeout as Timeout
from stopit import TimeoutException

matplotlib.use("agg")
from threading import Thread

import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

# import pyximport
# pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
# from c_digitize import c_digitize

import S2MSI
from S2MSI.Tools import ToolTip

numpy_types = {16: np.float16, 32: np.float32, 64: np.float64}

texts = {
    "welcome":
        'classical Bayesian for Sentinel-2: cB4S2 Version: %s' % __version__,
    "S2_MSI_granule_path":
        'Path or pattern to S2 granule folders which shall be processed. If not set otherwise by --output_directory, '
        'the output masks are written to the granule folders. This option is mutually incompatible with '
        '--tasks_input_file.',
    "tasks_input_file":
        'Path of input text file with one granule path per line. If not set otherwise by --output_directory, the output'
        'masks are written to the granule folders. This option is mutually incompatible with --tasks_input_file.',
    "END":
        'Done with processing, it was a pleasure serving you.',
    "TT_button_granules_file":
        'This will open a file selection dialog which allows to pick a single file. The selected file should contain '
        'only lines with a valid path on each line. Each path should point to a granule in a Sentinel-2 product. '
        'A possible example would be: \n '
        ' \n'
        '/[..]/S2A_[..].SAFE/GRANULE/S2A_[..]_T33VVC_N01.05 \n'
        '/[..]/S2A_[..].SAFE/GRANULE/S2A_[..]_T32VPH_N01.05 \n'
        '/[..]/S2A_[..].SAFE/GRANULE/S2A_[..]_T33UUB_N01.05 \n'
        ''
        'In this example, [..] denotes an arbitrary, but valid path on your file system.',
    "TT_button_granule_path":
        'This will open a folder selection dialog where a valid granule path for processing should be selected, '
        'A valid path could be [..]/S2A_[..].SAFE/GRANULE/S2A_[..]_T32VPH_N01.05, where [..] denotes an arbitrary, '
        'but valid path on the file system.',
    "TT_button_classifier_file":
        'The actual classifier is stored in a classifier file which is needed to operate the classification and '
        'which can be updated separately. Although a default value while we defined, a different file can be selected '
        'using this dialog.',
    "TT_button_output_folder":
        'Specify a output path where classification results can be stored. The current working directory'
        ' is set as default.',
    "TT_target_resolution":
        'Different Sentinel-2 MSI bands are distributed in 10m, 20m, and 60m spatial sampling. The detection is '
        'performed on a multi-spectral data set with homogeneous spatial sampling which is also the resulting '
        'resolution for the final mask.',
    "TT_interpolation_order":
        'Order of interpolation for homogenisation of different spatial samplings. 1: linear interpolation, '
        '2: quadratic interpolation, ....',
    "TT_number_of_threads":
        'The processor can run in parallel on the level of Sentinel-2 tiles (granules). Increasing the number of '
        'processes can decrease the total processing time but needs more main memory.',
    "TT_export_to_RGB":
        'Write jpeg image with RGB of Sentinel-2 image.',
    "TT_export_to_RGB_mask":
        'Write a gray scale jpeg image of the scene and blend with RGB view of the computed mask.',
    "TT_number_of_tiles":
        'The needed main memory per process depends on the fraction of the Sentinel-2 MSI image which is kept in '
        'memory. Increasing the number of tiles per image decreases the total amount of needed main memory.',
    "TT_export_confidence":
        'Export a image with the classification confidence for each pixel.',
    "TT_export_to_RGB_blend":
        'The computed mask is converted to a RGB color image and blended with a gray scale version of the original '
        'RGB image. The output is saved as jpeg file along with other results.',
    "TT_search_pattern":
        'Granules can be specified using a search pattern which is applied to all directories below the current '
        'working directory. The search is stopped after 10 seconds to prevent a frozen application. When using the '
        'GUI, the current pattern can be tested by hitting [ENTER] or clicking on the button to the right side. '
        'Current results will be printed to the output window.',
    "TT_export_jp2_csv":
        'Export mask to [jp2] image and store additional metadata in a separate csv file.',
    "TT_show_warnings":
        'As a default, warnings from python packages are only shown with this option set.',
    "TT_use_thread_pool":
        'Use thread pool instead of processes. Useful for debugging reasons to collect full error tracebacks.',
    "TT_RGB_channels":
        "Channels used to output a RGB image. Should be a comma separated list of 3 to 4 entry's. Possible choices"
        " are: 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'. If your "
        "channels are selected, the fourth one is interpreted as alpha channel."
}


def mask_image(args, S2_MSI_granule_path, logger=None):
    global S2_clf
    logger = logger or logging.getLogger(__name__)

    def fnly():
        try:
            return sys.stdout.lines, sys.stderr.lines
        except AttributeError:
            return None

    if args.output_directory == "":
        path_output = path.join(S2_MSI_granule_path, "IMG_DATA")
    else:
        path_output = args.output_directory

    if args.create_output_folder is True:
        S2MSI.Tools.mkdir_p(path_output)

    basename_output = path.basename(S2_MSI_granule_path)[:-7]

    outs = glob(path.join(path_output, basename_output + "*"))
    if len(outs) > 0:
        logger.info("Some Output already exists:")
        for out in outs:
            logger.info(out)

        if args.overwrite_output is True:
            logger.info("Continue overwriting existing files!")
        else:
            logger.info("Stop here since output already exists.")
            return fnly()

    import_bands = set(S2_clf.unique_channel_str)
    if args.export_RGB is True:
        import_bands = set(list(import_bands) + args.RGB_channels.split(","))

    t0 = time()
    S2_img = S2MSI.S2Image(S2_MSI_granule_path=S2_MSI_granule_path,
                           import_bands=import_bands,
                           data_mode="dense",
                           driver=args.jp2_driver,
                           target_resolution=args.target_resolution,
                           interpolation_order=args.interpolation_order
                           )
    logger.info("Total time to read S2 image data: %.2fs" % (time() - t0,))
    logger.info("Start detection.")
    t0 = time()
    S2_msk = S2_clf(S2_img)
    t1 = time()
    logger.info("Detection performed in %.2fs -> %.2f Mpx/s" % (
        t1 - t0, S2_img.data.shape[0] * S2_img.data.shape[1] / 10 ** 6 / (t1 - t0)))

    if args.export_RGB is True:
        rgb_img = S2_img.S2_image_to_rgb(rgb_bands=args.RGB_channels.split(","))
        fn = path.join(path_output, "%s_RGB.jpg" % basename_output)
        logger.info("Write RGB Image to: %s" % fn)
        S2_img.save_rgb_image(rgb_img=rgb_img, fn=fn)

    if args.export_confidence is True:
        fn_img = path.join(path_output, "%s_CONF.jp2" % basename_output)
        logger.info("Write output to: %s" % fn_img)
        S2_msk.export_confidence_to_jpeg2000(fn_img=fn_img)

    if args.export_mask_blend is True:
        rgb_img = S2_img.S2_image_to_rgb(rgb_bands=args.RGB_channels.split(","))
        fn = path.join(path_output, "%s_MASK_BLEND.jpg" % basename_output)
        logger.info("Write MASK with blended gray scale Image to: %s" % fn)
        S2_msk.export_mask_blend(fn_img=fn, rgb_img=rgb_img, alpha=args.alpha)
        del rgb_img

    if args.export_mask_rgb is True:
        rgb_img = S2_img.S2_image_to_rgb(rgb_bands=args.RGB_channels.split(","))
        fn = path.join(path_output, "%s_MASK_RGB.jpg" % basename_output)
        logger.info("Write MASK RGB Image to: %s" % fn)
        S2_msk.export_mask_rgb(fn_img=fn, rgb_img=rgb_img)
        del rgb_img

    if args.mask_export_format == "jp2":
        fn_img = path.join(path_output, "%s_MASK.jp2" % basename_output)
        fn_metadata = path.join(path_output, "%s.csv" % basename_output)
        logger.info("Write output to: %s" % fn_img)
        logger.info("Write output to: %s" % fn_metadata)
        S2_msk.export_to_jpeg200(fn_img=fn_img, fn_metadata=fn_metadata)
    else:
        raise ValueError("mask_export_format=%s is not understood" % args.mask_export_format)

    return fnly()


def main(args, logger=None):
    logger = logger or logging.getLogger(__name__)
    logger.info(texts["welcome"])

    if args.show_warnings is not True:
        logger.info("Switch off printing of warnings from python packages")
        warnings.filterwarnings("ignore")

    if args.persistence_file is not None:
        logger.info("Read classical Bayesian persistence file: %s" % args.persistence_file)
    builtins.S2_clf = S2MSI.CloudMask(logger=logger,persistence_file=args.persistence_file)

    if args.S2_MSI_granule_path is None:
        try:
            with Timeout(10.0, swallow_exc=False) as timeout_ctx:
                args.S2_MSI_granule_path = glob(args.glob_search_pattern, recursive=True)
        except TimeoutException:
            logger.info("Search for GRANULES was stoped after 10s without success -> stop here ")
            logger.info("The search pattern was: %s" % args.glob_search_pattern)
            return

        logger.info("No Input data given -> traverse local path and search for granules:")
        for granule in args.S2_MSI_granule_path:
            logger.info(granule)

    if args.number_of_threads == 1:
        logger.info("Start processing of %i jobs." % len(args.S2_MSI_granule_path))
        for granule in args.S2_MSI_granule_path:
            mask_image(args, granule)
    elif args.number_of_threads > 1:
        logger.info("Start processing of %i jobs using %i processes." % (
            len(args.S2_MSI_granule_path), args.number_of_threads))

        def _init_():
            globals()["update"] = lambda: None  # tkinter is not able to digest multiple threads
            sys.stdout = S2MSI.Tools.StdoutToList()
            sys.stderr = S2MSI.Tools.StdoutToList()

        logger.info("Start Processing in parallel running threads. Output of individual jobs is shown when done. ")

        if args.use_thread_pool is True:
            logger.info("!!!!!!!!!!!! Use ThreadPool  !!!!!!!!!")
            Pool = ThreadPool

        pool = Pool(initializer=_init_, processes=args.number_of_threads)
        tasks = [(args, granule, logger) for granule in args.S2_MSI_granule_path]
        jobs = [pool.apply_async(mask_image, task) for task in tasks]

        while len(jobs) > 0:
            sleep(5.0)
            readies = [job.ready() for job in jobs]
            for job, ready in zip(jobs, readies):
                if ready is True:
                    lne_out, lne_err = job.get()
                    logger.info("### print jop output ###")
                    for line in lne_out:
                        logger.info(line)
                    if len(lne_err) > 1:
                        logger.info("### errors occurred -> print stderr: ###")
                        for line in lne_out:
                            logger.info(line)
            jobs[:] = [job for job, ready in zip(jobs, readies) if ready is False]
            logger.info("#> Open Jobs: %i <#" % len(jobs))
        pool.close()

    else:
        raise ValueError("The number of threads should be larger or equal to zero and not:%i" % args.number_of_threads)

    logger.info(texts["END"])


# noinspection PyAttributeOutsideInit
class Gui(tk.Tk):
    def __init__(self, parent, args):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.args = args
        self.updateGUIThread = None
        self.init_gui()

        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

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

        ToolTip(self.tk_button1, text=texts["TT_button_granules_file"])

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
        lab = tk.Label(self, text="Search Pattern:")
        lab.grid(row=i_row, sticky=tk.E, column=0)
        frame = tk.Frame(self)
        frame.grid(row=i_row, column=1, sticky=tk.E + tk.W)
        self.tk_entry_search_pattern = tk.Entry(frame)
        self.tk_entry_search_pattern.insert(0, args.glob_search_pattern)
        self.tk_entry_search_pattern.pack(side=tk.LEFT, fill=tk.X, expand=1)
        self.tk_entry_search_pattern.bind('<Return>', self.test_pattern)
        self.tk_button5 = tk.Button(frame, text="Test Pattern", command=self.test_pattern)
        self.tk_button5.pack(side=tk.LEFT)
        ToolTip(self.tk_button5, text=texts["TT_search_pattern"])
        ToolTip(self.tk_entry_search_pattern, text=texts["TT_search_pattern"])
        ToolTip(lab, text=texts["TT_search_pattern"])

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
        for ii, (key, text, tt) in enumerate([("jp2", "Jpeg200+CSV", "TT_export_jp2_csv")]):
            bf = tk.Radiobutton(frame, text="%s" % text, variable=et, value=key)
            bf.pack(side=tk.LEFT, fill=tk.X)
            ToolTip(bf, text=texts[tt])

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
        out = S2MSI.Tools.StdRedirector(tk_text, gui=self, logfile=args.logfile_stub % args.suffix)
        sys.stdout = out
        sys.stderr = out

    def update_gui(self):
        while True:
            self.update()
            sleep(0.1)

    def main_gui(self):
        if self.updateGUIThread is None:
            self.updateGUIThread = Thread(target=self.update_gui)
            self.updateGUIThread.start()

        for button in self.buttons_deactivate_while_processing:
            button["state"] = "disabled"
        self.args.logging = True
        self.args.target_resolution = self.tr.get()
        self.args.number_of_threads = self.tk_scale.get()
        self.args.processing_tiles = self.tk_scale_tiles.get()
        self.args.interpolation_order = self.io.get()
        self.args.export_RGB = True if self.exp_to_rgb.get() == 1 else False
        self.args.export_confidence = True if self.exp_confidence.get() == 1 else False
        self.args.export_mask_blend = True if self.exp_to_rgb_mask.get() == 1 else False
        self.args.export_mask_rgb = True if self.exp_to_blend_mask.get() == 1 else False
        self.args.overwrite_output = True if self.oo.get() == 1 else False

        self.args.glob_search_pattern = self.tk_entry_search_pattern.get()

        print("Start with following settings:")
        for key, value in self.args.__dict__.items():
            print("%s -> %s" % (key, str(value)))

        try:
            main(args=self.args, logger=self.logger)
        except Exception as err:
            print("### Error occurred #####")
            print(err)
            print(repr(err))
            print(traceback.print_tb(err.__traceback__))
            pass

        for button in self.buttons_deactivate_while_processing:
            button["state"] = "normal"

    def test_pattern(self, event=None):
        pat = self.tk_entry_search_pattern.get()
        try:
            with Timeout(10.0, swallow_exc=False) as timeout_ctx:
                res = glob(pat, recursive=True)
        except TimeoutException:
            print("Search for GRANULES was stopped after 10s without success -> stop here ")
            print("The search pattern was: %s" % pat)
            return

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
            title='path to GRANULE folder, something like '
                  '*/S2A_OPER_PRD_[..].SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_[...]_N01.03/')
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

    major, minor, *_ = sys.version_info
    major_min, minor_min = 3, 5
    if (major, minor) < (major_min, minor_min):
        print("This program requires at least python %i.%i, this version is %i.%i. -> quit here." %
              (major_min, minor_min, major, minor))
        sys.exit(2)

    parser = argparse.ArgumentParser(prog='GFZ-detection',
                                     description='Cloud, Cirrus, Snow, Shadow Detection for Sentinel-2. ')

    group = parser.add_mutually_exclusive_group(required=False)

    def ad_ar(*args,**kwargs):
        parser.add_argument(*args,**kwargs)

    def ad_gr(*args,**kwargs):
        group.add_argument(*args,**kwargs)

    ad_gr("-i", "--S2_MSI_granule_path", action="store", type=str, nargs="*",help=texts["S2_MSI_granule_path"])
    ad_gr("-f", "--tasks_input_file", action="store", type=str, help=texts["tasks_input_file"])

    ad_ar("-r", "--target_resolution", help=texts["TT_target_resolution"], action="store", type=float,default=20.0, required=False)
    ad_ar("-o", "--interpolation_order", help=texts["TT_interpolation_order"], action="store", type=int,default=1, required=False, choices=range(1, 6))
    ad_ar("-p", "--persistence_file", help=texts["TT_button_classifier_file"], action="store", type=str,default="none")
    ad_ar("-e", "--mask_export_format", help=texts["TT_export_jp2_csv"], action="store", type=str,default="jp2", choices=["jp2"])
    ad_ar("-w", "--show_warnings", help=texts["TT_show_warnings"], action="store_true", default=False)
    ad_ar("-d", "--output_directory", help=texts["TT_button_output_folder"], action="store", type=str,default="./results/")
    ad_ar("-m", "--create_output_folder", help="", action="store_false", default=True)
    ad_ar("-n", "--number_of_threads", help=texts["TT_number_of_threads"], action="store", type=int,default=1)
    ad_ar("-q", "--use_thread_pool", help=texts["TT_use_thread_pool"], action="store_true", default=False)
    ad_ar("-j", "--RGB_channels", help=texts["TT_RGB_channels"], action="store", type=str,default="B11,B08,B03")
    ad_ar("-g", "--gui", help="asd", action="store", type=str, default="yes", choices=["yes", "no"])
    ad_ar("-x", "--export_RGB", help="asd", action="store_true", default=False)
    ad_ar("-z", "--export_mask_blend", help="TBD", action="store_true", default=False)
    ad_ar("-B", "--export_mask_rgb", help="TBD", action="store_true", default=False)
    ad_ar("-C", "--export_confidence", help="TBD", action="store_true", default=False)
    ad_ar("-T", "--processing_tiles", help="TBD", action="store", type=int, choices=range(0, 20),default=10)
    ad_ar("-D","--jp2_driver", help="TBD", action="store", type=str, default="gdal_JP2KAK", choices=["OpenJpeg2000", "gdal_JP2KAK"])

    ad_ar("-G", "--glob_search_pattern", help="TBD", action="store", type=str,default="./**/GRANULE/*S2A*")

    ad_ar("-W", "--overwrite_output", help="TBD", action="store_true", default=True)
    ad_ar("-l", "--logging", help="TBD", action="store_true", default=False)
    ad_ar("-L", "--logfile_stub", help="TBD", type=str, action="store", default="./cb4S2_%s.log")
    ad_ar("-A", "--alpha", help="TBD", action="store", type=float, default=0.6, required=False)

    args = parser.parse_args()

    tme = gmtime()
    args.suffix = "%i%i%i_%i:%i" % (tme.tm_year, tme.tm_mon, tme.tm_mday, tme.tm_hour, tme.tm_min)
    if args.persistence_file == "none":
        args.persistence_file = None

    if args.S2_MSI_granule_path is not None or args.tasks_input_file is not None or args.gui == "no":
        main(args)
    else:
        print("No command line argument were given -> start GUI")
        gui = Gui(parent=None, args=args)
        gui.mainloop()
