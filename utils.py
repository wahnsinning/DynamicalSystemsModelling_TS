<<<<<<< HEAD
# Re-import necessary modules after execution state reset
import os
import platform
import psutil
import subprocess
import multiprocessing


def print_cpu_specs():
    """
    Retrieves detailed CPU specifications relevant for multi-processing and multi-threading performance.
    Works across multiple operating systems: Windows, macOS, and Linux.
    """
    system_info = platform.system()

    # Get basic CPU information
    logical_cores = multiprocessing.cpu_count()
    physical_cores = psutil.cpu_count(logical=False)  # Only physical cores
    threads_per_core = logical_cores // physical_cores if physical_cores else "Unknown"

    # Get CPU clock speed
    if system_info == "Windows":
        try:
            cpu_freq = (
                float(subprocess.check_output("wmic cpu get MaxClockSpeed", shell=True).split()[1]) / 1000
            )  # MHz to GHz
        except Exception:
            cpu_freq = "Unknown"
    elif system_info == "Linux":
        try:
            cpu_freq = (
                float(subprocess.check_output("lscpu | grep 'MHz' | awk '{print $NF}'", shell=True).decode().strip())
                / 1000
            )
        except Exception:
            cpu_freq = "Unknown"
    elif system_info == "Darwin":  # macOS
        try:
            cpu_freq = float(subprocess.check_output("sysctl -n hw.cpufrequency", shell=True).decode().strip()) / 1e9
        except Exception:
            cpu_freq = "Unknown"
    else:
        cpu_freq = "Unknown"

    # CPU architecture
    architecture = platform.architecture()[0]

    # CPU name/model
    try:
        if system_info == "Windows":
            cpu_model = subprocess.check_output("wmic cpu get Name", shell=True).decode().split("\n")[1].strip()
        elif system_info == "Linux":
            cpu_model = (
                subprocess.check_output("lscpu | grep 'Model name' | awk -F: '{print $2}'", shell=True).decode().strip()
            )
        elif system_info == "Darwin":
            cpu_model = subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip()
        else:
            cpu_model = "Unknown"
    except Exception:
        cpu_model = "Unknown"

    # Get L1, L2, and L3 Cache sizes if available
    cache_sizes = {}
    if system_info == "Windows":
        cache_sizes["L1"] = "Unknown"
        cache_sizes["L2"] = "Unknown"
        cache_sizes["L3"] = "Unknown"  # Windows WMIC does not easily expose cache sizes
    elif system_info == "Linux":
        try:
            cache_sizes["L1"] = (
                subprocess.check_output("lscpu | grep 'L1d cache' | awk '{print $NF}'", shell=True).decode().strip()
            )
            cache_sizes["L2"] = (
                subprocess.check_output("lscpu | grep 'L2 cache' | awk '{print $NF}'", shell=True).decode().strip()
            )
            cache_sizes["L3"] = (
                subprocess.check_output("lscpu | grep 'L3 cache' | awk '{print $NF}'", shell=True).decode().strip()
            )
        except Exception:
            cache_sizes = {"L1": "Unknown", "L2": "Unknown", "L3": "Unknown"}
    elif system_info == "Darwin":
        try:
            cache_sizes["L1"] = (
                subprocess.check_output("sysctl -n hw.l1dcachesize", shell=True).decode().strip() + " bytes"
            )
            cache_sizes["L2"] = (
                subprocess.check_output("sysctl -n hw.l2cachesize", shell=True).decode().strip() + " bytes"
            )
            cache_sizes["L3"] = (
                subprocess.check_output("sysctl -n hw.l3cachesize", shell=True).decode().strip() + " bytes"
            )
        except Exception:
            cache_sizes = {"L1": "Unknown", "L2": "Unknown", "L3": "Unknown"}

    # Print results
    print("\n===== CPU Specifications =====")
    print(f"CPU Model: {cpu_model}")
    print(f"Architecture: {architecture}")
    print(f"Logical Cores: {logical_cores}")
    print(f"Physical Cores: {physical_cores}")
    print(f"Threads per Core: {threads_per_core}")
    print(f"CPU Frequency: {cpu_freq} GHz")
    print(f"L1 Cache: {cache_sizes.get('L1', 'Unknown')}")
    print(f"L2 Cache: {cache_sizes.get('L2', 'Unknown')}")
    print(f"L3 Cache: {cache_sizes.get('L3', 'Unknown')}")
    print("=============================\n")


if __name__ == "__main__":
    print_cpu_specs()
=======
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
>>>>>>> 12ee008436eeead1568f038fb2b93b70ca85ac39
