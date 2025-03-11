import os
import subprocess
from multiprocessing import cpu_count

def get_cpu_specs():
    # Checking system cpu cores 
    print(cpu_count())

    # Checking threads per core and CPU clock frequency

    # Get logical cores
    logical_cores = os.cpu_count()

    # Get physical cores
    physical_cores = int(subprocess.check_output("wmic cpu get NumberOfCores", shell=True).split()[1])

    # Calculate threads per core
    threads_per_core = logical_cores // physical_cores

    # Get CPU clock frequency
    cpu_frequency = float(subprocess.check_output("wmic cpu get MaxClockSpeed", shell=True).split()[1]) / 1000  # Convert MHz to GHz

    print(f"Logical cores: {logical_cores}")
    print(f"Physical cores: {physical_cores}")
    print(f"Threads per core: {threads_per_core}")
    print(f"CPU clock frequency: {cpu_frequency} GHz")