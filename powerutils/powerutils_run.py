# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
import numpy as np
from pathlib import Path
import annette.utils as utils
import annette.hardware.ncs2.parser as ncs2_parser
import matplotlib.pyplot as plt
from powerutils import __version__
from powerutils import processing

sys.path.append("./")


__author__ = "Matthias Wess"
__copyright__ = "Christian Doppler Laboratory for Embedded Machine Learning"
__license__ = "Apache 2.0"

_logger = logging.getLogger(__name__)


def extract(args):
    power_file= args.power_file
    power_path = args.power_path
    latency_file= args.latency_file
    latency_path = args.latency_path

    measured = ncs2_parser.extract_data_from_ncs2_report(latency_path, latency_path, latency_file, format="pickle")
    a_measured = utils.ncs2_to_format(measured)
    result, profile = processing.unite_latency_power_meas(a_measured , power_file, power_path, sample_rate = 500)
    return result, profile

def visualize(args):

    rate = args.rate 
    result, profile = extract(args)
    x= np.arange(len(profile))/rate
    f = plt.figure()
    plt.rcParams["figure.figsize"] = (8,2.5)
    plt.plot(x,profile, label='Power profile')
    n = 0
    for xc in np.cumsum(result['measured']):
        if n == 0:
            plt.axvline(x=xc,c='red',label='Layer transitions')
            n = 1
        else:
            plt.axvline(x=xc,c='red')

    plt.xlabel("Time [ms]")
    plt.ylabel("Power [W]")

    plt.legend()
    plt.show()


def parse_args(args):
    """Parse command line parameters.

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Powerutils Visualizer")
    parser.add_argument(
        "--version",
        action="version",
        version="powerutils {ver}".format(ver=__version__))
    parser.add_argument(
        "-r",
        "--rate",
        dest="rate",
        default=500,
        help="sampling rate in kHz",
        type=int,
        metavar="in")
    parser.add_argument(
        "-lf",
        "--lfile",
        dest="latency_file",
        default=None,
        help="latency filename",
        type=str,
        metavar="in")
    parser.add_argument(
        "-lp",
        "--lpath",
        dest="latency_path",
        default=None,
        help="latency path",
        type=str,
        metavar="in")
    parser.add_argument(
        "-pf",
        "--pfile",
        dest="power_file",
        default=None,
        help="power filename",
        type=str,
        metavar="in")
    parser.add_argument(
        "-pp",
        "--ppath",
        dest="power_path",
        default=None,
        help="power path",
        type=str,
        metavar="in")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging.

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls.

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    visualize(args)

def run():
    """Entry point for console_scripts."""
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
