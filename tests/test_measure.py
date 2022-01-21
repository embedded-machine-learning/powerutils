# port 0 - NCS2
# port 1 - Jetson TX2
# port 2 - Edge TPU
# port 3 - ZCU102
# port 4 - Raspberry Pi 4
# port 5 - TinyML platforms
# future ports reserved for: Jetson Xavier, Jetson Nano, Pybadge, Raspberry Pi

from powerutils import measurement
import time, os, paramiko

def test_power():
    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60, port=0)

    #print(pm.__dict__)
    test_kwargs = {"hardware": "test"}

    pm.start_gather(test_kwargs)

    try:
        for i in range(300):
            time.sleep(1)
            print("seconds elapsed:", i, flush=True, end="\r")
    except KeyboardInterrupt:
        print("sleeping loop exited")

    pm.end_gather(True) # stop the power measurement

    assert True, "power measurement passed"

def test_rpi4():
    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./rpi4_desktop", max_duration=60, port=4)

    #print(pm.__dict__)
    test_kwargs = {"hardware": "rpi4", "os": "ubuntu_desktop_aarch64", "model_name": "squeezenet"}

    pm.start_gather(test_kwargs)

    try:
        for i in range(300):
            time.sleep(1)
            print("seconds elapsed:", i, flush=True, end="\r")
    except KeyboardInterrupt:
        print("sleeping loop exited")

    pm.end_gather(True) # stop the power measurement

    assert True, "power measurement passed"

def test_rpi4_ssh():
    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60, port=4)

    # execute script on remote machine
    ssh_ip = "192.168.0.100"
    ssh_username = "ubuntu"
    ssh_command_execute_nets = "python3 /home/ubuntu/pyarmnn/tflite_pyarmnn_rand_inputs_v2.py"

    # initialize and connect ssh
    client = paramiko.SSHClient()
    print("setup client")
    client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy)
    print("set host policy")
    client.connect(ssh_ip, port=22, username=ssh_username, password="raspberry")
    print("connected")
    stdin, stdout, stderr = client.exec_command("ls /home/ubuntu/pyarmnn/")  # execute remote command
    print(stdout)

    for m in modelnames:
        test_kwargs = {"hardware": "rpi4_ubuntu_server_aarch64", "model_name": modelname}
        pm.start_gather(test_kwargs)
        stdin, stdout, stderr = client.exec_command(ssh_command_execute_nets)  # execute remote command

        # print command results
        for line in stdout:
            try:
                print('... ' + line.strip('\n'))
            except KeyboardInterrupt:
                print("exited on line:", line)

    client.close()  # close the ssh channel

    pm.end_gather(True) # stop the power measurement

    assert True, "power measurement passed"

def test_cf_inceptionv1_imagenet_224_224():
    print("Inception v1 measurement")

    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60)
    test_kwargs = {"model_name": "cf_inceptionv1_imagenet_224_224", "index_run": None, "api": "async", "niter": 10, "nireq": 1, "batch": 1}
    pm.start_gather(test_kwargs)

    bench_app_file = "/opt/intel/openvino_2021.3.394/deployment_tools/tools/benchmark_tool/benchmark_app.py"
    xml_path = "~/powermeas/annet_models_pb/cf_inceptionv1_imagenet_224_224_3.16G/converted/deploy.xml"

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
    xml_path = "~/powermeas/annet_models_pb/tf_ssd_voc_300_300_64.81G/converted/frozen.xml"

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

def test_coral_mobilenetv1(query):
    print("EDGE TPU measurement")

    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60, port=2)
    test_kwargs = {"model_name": "coral_test_mobilenetv1_224", "query": query}
    pm.start_gather(test_kwargs)

    c_bench = ("python3 /home/intel-nuc/mlperf/vision/classification_and_detection/python/main.py " +
    "--model /home/intel-nuc/powermeas/coral/coral_models/mobilenet_v1_1.0_224_quant_edgetpu.tflite " +
    "--dataset-path /home/intel-nuc/powermeas/coral/coral_dataset/ILSVRC2012_img_val " +
    "--profile mobilenet_coral " +
    "--mlperf_conf /home/intel-nuc/mlperf/mlperf.conf " +
    "--user_conf /home/intel-nuc/mlperf/vision/classification_and_detection/user.conf " +
    "--accuracy --count 10 " +
    "--samples-per-query " + str(query))

    # start inference
    if os.system(c_bench):
        print("An error has occured during benchmarking!")

    pm.end_gather(True)  # stop the power measurement

    assert True, "power measurement of Coral passed"

def test_coral_inceptionv1(query):
    print("EDGE TPU measurement")

    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60, port=2)
    test_kwargs = {"model_name": "coral_test_inceptionv1_224", "query": query}
    pm.start_gather(test_kwargs)

    c_bench = ("python3 /home/intel-nuc/mlperf/vision/classification_and_detection/python/main.py " +
    "--model /home/intel-nuc/powermeas/coral/coral_models/inception_v1_224_quant_edgetpu.tflite " +
    "--dataset-path /home/intel-nuc/powermeas/coral/coral_dataset/ILSVRC2012_img_val " +
    "--profile mobilenet_coral " +
    "--mlperf_conf /home/intel-nuc/mlperf/mlperf.conf " +
    "--user_conf /home/intel-nuc/mlperf/vision/classification_and_detection/user.conf " +
    "--accuracy --count 10 "
    "--samples-per-query " + str(query))

    # start inference
    if os.system(c_bench):
        print("An error has occured during benchmarking!")

    pm.end_gather(True)  # stop the power measurement

    assert True, "power measurement of Coral passed"

def test_coral_all():
    queries = [1, 2, 4, 8, 16, 32, 64, 128]
    for q in queries:
        test_coral_mobilenetv1(q)
        test_coral_inceptionv1(q)

def test_inception_v1_224_quant():
    print("inception_v1_224_quant measurement")

    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60)
    test_kwargs = {"model_name": "inception_v1_224_quant", "index_run": None, "api": "async", "niter": 10, "nireq": 1, "batch": 1}
    pm.start_gather(test_kwargs)

    bench_app_file = "/opt/intel/openvino_2021.3.394/deployment_tools/tools/benchmark_tool/benchmark_app.py"
    xml_path = "/home/intel-nuc/powermeas/coral/coral_models/inception_v1_224_quant_20181026/tmp/inception_v1_224_quant_frozen.xml"

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

    assert True, "power measurement of inception_v1_224_quant passed"

def test_tx2():
    print("testing power measurements on the TX2")
    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=1000, port=1)
    test_kwargs = {"model_name": "tx2", "info": "evaltrt"}
    pm.start_gather(test_kwargs)

    # execute script on remote machine
    ssh_ip = "192.168.1.13"
    ssh_username = "cdleml"
    ssh_command_stress = "stress -c 6 -t 2 -v"
    ssh_command_runOnTX2 = "/media/cdleml/128GB/Users/lsteindl/masterthesis/runOnTx2_reduced.sh"

    # initialize and connect ssh
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy)
    client.connect(ssh_ip, username=ssh_username)

    stdin, stdout, stderr = client.exec_command(ssh_command_runOnTX2) # execute remote command

    # print command results
    for line in stdout:
        try:
            print('... ' + line.strip('\n'))
        except KeyboardInterrupt:
            print("exited on line:", line)

    client.close()  # close the ssh channel
    pm.end_gather(True)  # stop the power measurement

    assert True, "power measurement of TX2 passed"

def test_zcu102():
    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60, port=3)

    test_kwargs = {"model_name": "zcu102", "test": "test2"}

    pm.start_gather(test_kwargs)

    try:
        for i in range(300):
            time.sleep(1)
            print("seconds elapsed:", i, flush=True, end="\r")
    except KeyboardInterrupt:
        print("sleeping loop exited")

    pm.end_gather(True)  # stop the power measurement

    assert True, "power measurement passed"

def test_tinyML():
    pm = measurement.power_measurement(sampling_rate=500000, data_dir="./tmp", max_duration=60, port=5)

    #print(pm.__dict__)
    test_kwargs = {"hardware": "tinyML_5Vbipol"}

    pm.start_gather(test_kwargs)

    try:
        for i in range(300):
            time.sleep(1)
            print("seconds elapsed:", i, flush=True, end="\r")
    except KeyboardInterrupt:
        print("sleeping loop exited")

    pm.end_gather(True) # stop the power measurement

    assert True, "power measurement passed"


def main():
    #test_power()
    #test_cf_inceptionv1_imagenet_224_224()
    #test_cf_tf_ssd_voc_300_300()
    #test_coral_all()
    #test_inception_v1_224_quant()
    #test_tx2()
    #test_coral_mobilenetv1(1)
    #test_coral_inceptionv1(1)
    #test_zcu102()
    #test_rpi4()
    test_tinyML()

if __name__ == "__main__":
    # run main
    main()

