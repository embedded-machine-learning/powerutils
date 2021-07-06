from powerutils import measurement
import time

def test_power():
    pm = measurement.power_measurement(sampling_rate=500000, data_dir="tests/tmp", max_duration=60)

    # print(pm.__dict__)
    test_kwargs =  {"model_name": "vgg16"}

    pm.start_gather(test_kwargs)
    print("do something")
    time.sleep(2)
    pm.end_gather(True) # stop the power measurement

    assert True, "power measurement passed"

def main():
    test_power()

if __name__ == "__main__":
    main()
