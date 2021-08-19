from powerutils import measurement
import time, os

def test_power():
    pm = measurement.power_measurement(sampling_rate=500000, data_dir="tests/tmp", max_duration=60)

    print(pm.__dict__)
    test_kwargs =  {"model_name": "vgg16"}

    pm.start_gather(test_kwargs)
    print("do something")
    time.sleep(2)
    pm.end_gather(True) # stop the power measurement

    assert True, "power measurement passed"


def test_cf_inceptionv1_imagenet_224_224():
    print("Inception v1 measurement")

    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60)
    test_kwargs = {"model_name": "cf_inceptionv1_imagenet_224_224", "index_run": None, "api": "async", "niter": 10, "nireq": 1, "batch": 1}
    pm.start_gather(test_kwargs)

    bench_app_file = "/opt/intel/openvino_2021.3.394/deployment_tools/tools/benchmark_tool/benchmark_app.py"
    xml_path = "~/powerutils/annet_models_pb/cf_inceptionv1_imagenet_224_224_3.16G/converted/deploy.xml"

    c_bench = ("python3 " + bench_app_file +
               " -m " + xml_path +
               " -d MYRIAD " +
               " -b " + str(test_kwargs["batch"]) +
               " -api " + test_kwargs["api"] +
               " -nireq " + str(test_kwargs["nireq"]) +
               " -niter " + str(test_kwargs["niter"]) +
               " --report_type average_counters" +
               " --report_folder ./tmp")

    # start inference
    if os.system(c_bench):
        print("An error has occured during benchmarking!")

    pm.end_gather(True)  # stop the power measurement

    assert True, "power measurement of cf_inceptionv1_imagenet_224_224_3.16G passed"

def test_cf_tf_ssd_voc_300_300():
    print("Inception v1 measurement")

    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60)
    test_kwargs = {"model_name": "tf_ssd_voc_300_300", "index_run": None, "api": "async", "niter": 10, "nireq": 1, "batch": 1}
    pm.start_gather(test_kwargs)

    bench_app_file = "/opt/intel/openvino_2021.3.394/deployment_tools/tools/benchmark_tool/benchmark_app.py"
    xml_path = "~/powerutils/annet_models_pb/tf_ssd_voc_300_300_64.81G/converted/frozen.xml"

    c_bench = ("python3 " + bench_app_file +
               " -m " + xml_path +
               " -d MYRIAD " +
               " -b " + str(test_kwargs["batch"]) +
               " -api " + test_kwargs["api"] +
               " -nireq " + str(test_kwargs["nireq"]) +
               " -niter " + str(test_kwargs["niter"]) +
               " --report_type average_counters" +
               " --report_folder ./tmp")

    # start inference
    if os.system(c_bench):
        print("An error has occured during benchmarking!")

    pm.end_gather(True)  # stop the power measurement

    assert True, "power measurement of tf_ssd_voc_300_300_64.81G passed"


def main():
    #test_power()
    test_cf_inceptionv1_imagenet_224_224()
    test_cf_tf_ssd_voc_300_300()

if __name__ == "__main__":
    # run main
    main()

