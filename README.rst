powerutils
==========

.. image:: https://img.shields.io/pypi/v/powerutils.svg
    :target: https://pypi.python.org/pypi/powerutils
    :alt: Latest PyPI version

.. image:: https://travis-ci.org/kragniz/cookiecutter-pypackage-minimal.png
   :target: https://travis-ci.org/kragniz/cookiecutter-pypackage-minimal
   :alt: Latest Travis CI build status

Power Utility Tools for DNN analysis

Usage
-----

Gathering data via predefined test cases in console (see script for available options)::

    python3 ./tests/test_measure.py

Instantiation in code::

    from powerutils import measurement

    # create instance of class
    pm = measurement.power_measurement(
        sampling_rate=500000, # set the sampling rate to whatever your device supports
        data_dir = "./tmp", # pass the folder where data files will be saved
        max_duration=60, # set the maximal duration of the gathering process [seconds]
        port=0, # if your DAQ device has more than one port, choose the port where the DUT is connected to
        range_index=3) # choose a measurement resolution from available ranges (the higher, the more accurate)

    # define parameters for the name of the data file
    test_kwargs = {"model_name" : "awesome_model", "index_run" : 1, "my_parameter" : "some_value"}

    pm.start_gather(test_kwargs) # start the data aquisition

    # here should be the inference on a platform
    from time import sleep; sleep(2); # or a sleep command to test the data gathering

    pm.end_gather(True) # ends the data gathering and writes it to a data (.dat) file
    print("Finished")

Installation
------------

Clone the repository to your machine and navigate into it::

    git clone https://github.com/embedded-machine-learning/powerutils.git
    cd powerutils

(OPTIONAL) Create a Python3 virtual environment and activate it::

    python3 -m venv venv_powerutils
    source venv_powerutils/bin/activate

Install powerutils locally and check the installation::

    pip3 install -e .
    python3 -c "import powerutils; help(powerutils)"

The last command should show general information of the module. Exit help() by typing "q" (without the quotation marks)

Requirements
^^^^^^^^^^^^

Linux machine with Python3 installed (tested on Ubuntu 18.04 LTS)
A Data Aqcuisition Card from `https://www.mccdaq.com`
Python3 modules: uldaq, numpy, pandas, matplotlib
OpenVino for the profiling of the Intel Neural Compute Stick 2
TF Lite for profiling of the Google Edge TPU

Compatibility
-------------

Licence
-------

Authors
-------

`powerutils` was written by `CDL EML <cdleml@tuwien.ac.at>`_.
