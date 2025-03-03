import logging
import subprocess

module_logger = logging.getLogger("icad_transcribe.gpu")

# -------------
# GPU Handlers
# -------------
def get_available_gpus():
    """
    Returns the number of available GPUs by calling 'nvidia-smi --list-gpus'.
    Logs errors and re-raises exceptions so the caller can handle them.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            gpu_list = result.stdout.strip().split("\n")
            return len(gpu_list)
        else:
            error_msg = f"Error running nvidia-smi: {result.stderr}"
            module_logger.error(error_msg)
            # You can decide to return 0 or raise an exception. Here's an example of raising:
            raise RuntimeError(error_msg)
    except Exception as e:
        module_logger.error(f"Exception occurred while detecting GPUs: {e}")
        raise

def get_gpu_memory(gpu_index):
    """
    Returns the amount of free memory (in MB) for the given GPU index.
    Logs errors and re-raises exceptions so the caller can handle them.
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.free',
                '--format=csv,nounits,noheader',
                '-i', str(gpu_index)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            available_memory = int(result.stdout.strip())  # Memory in MB
            return available_memory
        else:
            error_msg = f"Error running nvidia-smi for GPU {gpu_index}: {result.stderr}"
            module_logger.error(error_msg)
            raise RuntimeError(error_msg)
    except Exception as e:
        module_logger.error(f"Exception occurred while getting memory for GPU {gpu_index}: {e}")
        raise

def get_gpu_utilization(gpu_index):
    """
    Returns the GPU utilization (in percentage) for the given GPU index.
    Logs errors and re-raises exceptions so the caller can handle them.
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu',
                '--format=csv,nounits,noheader',
                '-i', str(gpu_index)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            utilization = int(result.stdout.strip())  # Utilization in %
            return utilization
        else:
            error_msg = f"Error running nvidia-smi for GPU {gpu_index}: {result.stderr}"
            module_logger.error(error_msg)
            raise RuntimeError(error_msg)
    except Exception as e:
        module_logger.error(f"Exception occurred while getting utilization for GPU {gpu_index}: {e}")
        raise
