#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from psutil._common import shwtemp

from corepycler.cli import print_cpu_temperature

CPU_INTEL_TEMPERATURES = {'acpitz': [shwtemp(label='', current=91.0, high=103.0, critical=103.0)], 'nvme': [shwtemp(label='Composite', current=56.85, high=89.85, critical=94.85), shwtemp(label='Sensor 1', current=56.85, high=65261.85, critical=65261.85), shwtemp(label='Sensor 2', current=46.85, high=65261.85, critical=65261.85)], 'coretemp': [shwtemp(label='Package id 0', current=89.0, high=100.0, critical=100.0), shwtemp(label='Core 14', current=87.0, high=100.0, critical=100.0), shwtemp(label='Core 15', current=85.0, high=100.0, critical=100.0), shwtemp(label='Core 0', current=85.0, high=100.0, critical=100.0), shwtemp(label='Core 4', current=87.0, high=100.0, critical=100.0), shwtemp(label='Core 8', current=89.0, high=100.0, critical=100.0), shwtemp(label='Core 9', current=89.0, high=100.0, critical=100.0), shwtemp(label='Core 10', current=89.0, high=100.0, critical=100.0), shwtemp(label='Core 11', current=89.0, high=100.0, critical=100.0), shwtemp(label='Core 12', current=87.0, high=100.0, critical=100.0), shwtemp(label='Core 13', current=85.0, high=100.0, critical=100.0)], 'iwlwifi_1': [shwtemp(label='', current=46.0, high=None, critical=None)]}
CPU_AMD_TEMPERATURES = {'nvme': [shwtemp(label='Composite', current=28.85, high=83.85, critical=87.85)], 'k10temp': [shwtemp(label='Tctl', current=45.875, high=None, critical=None), shwtemp(label='Tccd1', current=33.5, high=None, critical=None), shwtemp(label='Tccd2', current=33.25, high=None, critical=None)], 'spd5118': [shwtemp(label='', current=30.75, high=55.0, critical=85.0)], 'amdgpu': [shwtemp(label='edge', current=38.0, high=None, critical=None)]}
CPU_ARM_TEMPERATURES = {'cpu_thermal': [shwtemp(label='', current=36.511, high=None, critical=None)]}


def test_print_cpu_temperature(mocker):
    sensors_temperatures_mock = mocker.patch('psutil.sensors_temperatures')

    sensors_temperatures_mock.return_value = CPU_INTEL_TEMPERATURES
    print_cpu_temperature()

    sensors_temperatures_mock.return_value = CPU_AMD_TEMPERATURES
    print_cpu_temperature()

    sensors_temperatures_mock.return_value = CPU_ARM_TEMPERATURES
    print_cpu_temperature()