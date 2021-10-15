# this script takes a neural network in the intermediate representation .pd
# and converts it to a Movidius NCS2 conform format with .xml and .bin
# runs inference on the generated model

import argparse
import os, sys, threading
from os import system
from sys import stdout
from time import sleep, time
import numpy as np


def run_bench(daq_device, low_channel, high_channel, input_mode,ranges, samples_per_channel, rate, scan_options, flags,
              data, data_dir, data_fname,  power_measurement, index_pm,
              xml = "", model = "", save_folder = "./tmp", report_dir = "report", niter = 10, api = "async", proto="", nireq=1, batch=1):
    global bench_over
    mo_file = os.path.join("/", "opt", "intel", "openvino",
    "deployment_tools", "model_optimizer", "mo.py")
    bench_app_file = os.path.join("/","opt","intel", "openvino",
    "deployment_tools", "tools", "benchmark_tool", "benchmark_app.py")

    # check if necessary files exists
    if not os.path.isfile(mo_file):
        print("model optimizer not found at:", mo_file)

    if not os.path.isfile(bench_app_file):
        print("benchmark_app not found at:", bench_app_file)

    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    # set framework string and model name deploy.pb/forzen.pb
    framework = ""
    if "tf_" in model:
        framework = "tf_"
        default_name = "frozen."
    elif "cf_" in model:
        framework = "cf_"
        default_name = "deploy."
    elif "dk_" in model:
        framework = "dk_"
        default_name = "deploy."
    elif "ox_" in model:
        framework = "ox_"
        default_name = "default."

    # if no .pb is given look if an .xml already exists and take it
    # if no .pb or .xml is given exit!
    print("\n**********Movidius FP16 conversion**********")
    xml_path = os.path.join(save_folder, framework + model.split(framework)[1].split("/")[0] + "_b" + str(batch) + ".xml")
    model_name = ""

    if os.path.isfile(xml_path):
        print("Using already converted model!", xml_path)
    elif os.path.isfile(model):
        # yolov3/yolov3-tiny json file necessary for conversion
        conv_cmd_str = ""
        if "yolov3-tiny" in model or "yolov3-tiny" in xml :
            conv_cmd_str = (" --transformations_config" +
            " /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3_tiny.json")
        elif "yolov3" in model or "yolov3-tiny" in xml :
            conv_cmd_str = (" --transformations_config" +
            " /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json")

        if "tf_" in model:
            # Tensorflow conversion
            # input_shape for tensorflow : batch, width, height, channels
            shape = "[" + str(batch) + "," + model.split("tf_")[1].split("_")[2] + "," +  model.split("tf_")[1].split("_")[3]+ ",3]"

            c_conv = ("python3 " + mo_file +
            " --input_model " + model +
            " --output_dir " + save_folder +
            " --data_type FP16 " +
            " --input_shape " + shape +
            conv_cmd_str)
            xml_path = os.path.join(save_folder, model.split(".pb")[0].split("/")[-1]+".xml")
        elif "cf_" in model or "dk_" in model:
            # Caffe or Darknet conversion
            # input_shape : batch, channels, width, height
            input_proto =  model.split("/deploy.caffemodel")[0] + "/deploy.prototxt"
            if "cf_" in model:
                shape = "[" + str(batch) + ",3," + model.split("cf_")[1].split("_")[2] + "," + model.split("cf_")[1].split("_")[3] + "]"
            elif "dk_" in model:
                shape = "[" + str(batch) + ","+ model.split("dk_")[1].split("_")[2] + "," + model.split("dk_")[1].split("_")[3] + ",3]"

            if "SPnet" in model:
                input_node = "demo"
            else:
                input_node = "data"

            if "yolov3" in model or "yolov3-tiny" in model:
                input_node = "inputs"

            c_conv = ("python3 " + mo_file +
            " --input_model " + model +
            #" --input_proto " + input_proto +
            " --output_dir " + save_folder +
            " --data_type FP16 " +
            " --input_shape " + shape +
            " --input " + input_node + # input node sometimes called demo
            conv_cmd_str)
        elif "ox_" in model:
            shape = "[" + str(batch) + ",3," + model.split("ox_")[1].split("_")[2] + "," + \
                    model.split("ox_")[1].split("_")[3] + "]"

            c_conv = ("python3 " + mo_file +
                      " --input_model " + model +
                      # " --input_proto " + input_proto +
                      " --output_dir " + save_folder +
                      " --data_type FP16 " +
                      " --input_shape " + shape +
                      " --keep_shape_ops " +
                      " --input data" +
                      conv_cmd_str)

        if os.system(c_conv):
            print(c_conv)
            print("\nAn error has occured during conversion!\n")

        # rename all three generated files
        extension_list = ["xml", "bin", "mapping"]
        for ex in extension_list:
            os.rename(os.path.join(save_folder, default_name + ex),
            os.path.join(save_folder, framework + model.split(framework)[1].split("/")[0] + "_b" + str(batch) + "." + ex))

        xml_path = os.path.join(save_folder, framework + model.split(framework)[1].split("/")[0] + "_b" + str(batch) + ".xml")

    model_name = xml_path.split(".xml")[0].split("/")[-1]

    c_bench = ("python3 " + bench_app_file +
    " -m "  + xml_path +
    " -d MYRIAD " +
    " -b " + str(batch) +
    " -api " + api +
    " -nireq " + str(nireq) +
    " -niter " + str(niter) +
    " --report_type average_counters" +
    " --report_folder " + report_dir)

    # start measurement in parallel to inference
    #daq_measurement(low_channel, high_channel, input_mode,ranges, samples_per_channel, rate, scan_options, flags, data)

    if uldaq_import and power_measurement == "True":
        print("Starting power measurement")
        x = threading.Thread(target=daq_measurement, args=(daq_device, low_channel, high_channel, input_mode,
                        ranges, samples_per_channel,
                        rate, scan_options, flags, data, data_dir, data_fname, index_pm, api, niter, nireq, model_name, batch))
        x.start()

    try:
        # start inference
        if os.system(c_bench):
            print("An error has occured during benchmarking!")
    except KeyboardInterrupt:
        print("Keyboard Interrupt caught, ending benchmark_app...")

    new_avg_bench_path = os.path.join(report_dir, "_".join(("bacr", model_name.split(".pb")[0], str(index_pm), api,
                                                          "n" + str(niter), "ni" + str(nireq), "b" + str(batch) + ".csv")))
    new_stat_rep_path = os.path.join(report_dir, "_".join(("stat_rep", model_name.split(".pb")[0], str(index_pm), api,
                                                           "n" + str(niter), "ni" + str(nireq), "b" + str(batch) + ".csv")))
    # rename the default report file name
    if os.path.isfile(os.path.join(report_dir, "benchmark_average_counters_report.csv")):
        os.rename(os.path.join(report_dir, "benchmark_average_counters_report.csv"), new_avg_bench_path)
    if os.path.isfile(os.path.join(report_dir, "benchmark_report.csv")):
        os.rename(os.path.join(report_dir, "benchmark_report.csv"), new_stat_rep_path)

    bench_over = True # this ends the power data gathering
    print("**********REPORTS GATHERED**********")
    print("Average Counters:", new_avg_bench_path)
    print("Statistics:", new_stat_rep_path)

    return new_avg_bench_path, new_stat_rep_path


def main1(args, b=None, ni=None, a=None):
    if not args.model and not args.xml:
        sys.exit("Please pass either a frozen pb or IR xml/bin model")

    if uldaq_import and args.power_measurement == "True":
        base = 500000
        daq_device, low_channel, high_channel, input_mode, meas_range, \
        samples_per_channel,  rate, scan_options, flags, data, data_dir, data_fname = daq_setup(base,base*240)
    else:
        daq_device, low_channel, high_channel, input_mode, meas_range, \
        samples_per_channel, rate, scan_options, flags, data, data_dir, data_fname = None, None, None, None, None, None, \
                                                                                     None, None, None, None, None, None,
    if ni is not None:
        nireq = ni
    else:
        nireq = args.nireq

    if a is not None:
        api = a
    else:
        api = args.api

    if b is not None:
        batch = b
    else:
        batch = args.batch

    print("Working on - batch:", batch, " with:", nireq, "inference requests")

    run_bench(daq_device, low_channel, high_channel, input_mode, meas_range, samples_per_channel, rate, scan_options, flags,
              data, data_dir, data_fname,  args.power_measurement, index_run,
              xml=args.xml, model=args.model, save_folder=args.save_folder,
              report_dir=args.report_dir, niter=args.niter,
              api=api, proto=args.proto, nireq=nireq, batch=batch)



if __name__ == "__main__":

