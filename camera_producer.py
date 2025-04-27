# camera_producer.py
# Captures frames and writes them to shared memory.

#==========<NECESSARY PACKAGES>================
# desc:  python3.11 / picamera / numpy / posix_ipc
#
#==========<ACTIVATE_1st_VIRTUAL_ENVIRONMENT_IN_NEW_TERMINAL>===================
# $ source picamera_env/bin/activate
#
#===========<RUN_THIS_SCRIPT_FIRST>================
# python /home/TrashBot/yolo_IPC/camera_producer.py


import numpy as np
import time
import sys
from multiprocessing import shared_memory
import posix_ipc # Use POSIX IPC for semaphores
import signal
import errno # For checking semaphore errors
from picamera2 import Picamera2

# --- Configuration (MUST MATCH CONSUMER) ---
FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS = 640, 480, 3
FRAME_DTYPE = np.uint8
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS * np.dtype(FRAME_DTYPE).itemsize
SHM_NAME = 'psm_frame_buffer'
# Use POSIX-style names for semaphores (start with /)
FRAME_READY_SEM_NAME = '/psm_frame_ready_sem' # Flag: Producer -> Consumer
CONSUMER_READY_SEM_NAME = '/psm_consumer_ready_sem' # Flag: Consumer -> Producer
# --- End Configuration ---

# --- Global variables for cleanup ---
shm = None
picam2 = None
frame_ready_sem = None
consumer_ready_sem = None
running = True

def cleanup(signum, frame):
    global running
    print("Camera Producer: Signal received, initiating cleanup...")
    running = False

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print("Camera Producer: Starting...")

try:
    # --- Create Shared Memory ---
    try:
        shm = shared_memory.SharedMemory(create=True, size=FRAME_SIZE, name=SHM_NAME)
        print(f"Camera Producer: Created shared memory block '{SHM_NAME}'")
    except FileExistsError:
        print(f"Camera Producer: Shared memory block '{SHM_NAME}' exists. Attaching...")
        shm = shared_memory.SharedMemory(name=SHM_NAME)

    # --- Create/Initialize Named Semaphores ---
    # O_CREAT creates if it doesn't exist. Initialize with 0 or 1.
    # Use try-except to handle potential cleanup issues from previous runs
    try:
        # Frame Ready: Initial value 0 (producer signals by releasing)
        frame_ready_sem = posix_ipc.Semaphore(FRAME_READY_SEM_NAME, posix_ipc.O_CREAT, initial_value=0)
        print(f"Camera Producer: Opened/Created semaphore '{FRAME_READY_SEM_NAME}'")
        # Ensure value is 0 if it already existed and might be > 0
        while frame_ready_sem.value > 0:
             frame_ready_sem.acquire() # Lower value to 0
        print(f"Camera Producer: Semaphore '{FRAME_READY_SEM_NAME}' initial value confirmed 0.")

    except Exception as e:
         print(f"Error creating/opening frame_ready semaphore: {e}")
         raise # Re-raise to stop execution

    try:
        # Consumer Ready: Initial value 1 (consumer signals by releasing, producer waits by acquiring)
        consumer_ready_sem = posix_ipc.Semaphore(CONSUMER_READY_SEM_NAME, posix_ipc.O_CREAT, initial_value=1)
        print(f"Camera Producer: Opened/Created semaphore '{CONSUMER_READY_SEM_NAME}'")
         # Ensure value is 1 if it already existed and might be != 1
        while consumer_ready_sem.value > 1:
             consumer_ready_sem.acquire() # Lower value
        while consumer_ready_sem.value < 1:
             consumer_ready_sem.release() # Raise value
        print(f"Camera Producer: Semaphore '{CONSUMER_READY_SEM_NAME}' initial value confirmed 1.")

    except Exception as e:
         print(f"Error creating/opening consumer_ready semaphore: {e}")
         raise # Re-raise to stop execution


    # Attach numpy array
    shared_array = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=FRAME_DTYPE, buffer=shm.buf)
    print("Camera Producer: Attached NumPy array to shared memory.")

    # --- Setup Picamera2 ---
    print("Camera Producer: Initializing Picamera2...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    print("Camera Producer: Picamera2 started. Warming up...")
    time.sleep(2)

    # --- Main Loop ---
    print("Camera Producer: Starting capture loop...")
    while running:
        # 1. Wait for the consumer to signal it's ready (acquire consumer_ready_sem)
        # This blocks until the semaphore value > 0, then decrements it.
        print(f"Camera Producer: Waiting for consumer ready signal (sem value: {consumer_ready_sem.value})...")
        try:
            # Use timed acquire to prevent indefinite block if consumer dies
            consumer_ready_sem.acquire(timeout=2.0) # Wait up to 2 seconds
        except posix_ipc.BusyError: # This is the exception for timeout
            if not running: break # Exit if cleanup signal received
            print("Camera Producer: Timed out waiting for consumer ready signal. Checking again...")
            continue
        except Exception as e:
             print(f"Camera Producer: Error acquiring consumer_ready_sem: {e}")
             running = False # Stop on error
             break

        print(f"Camera Producer: Consumer ready signal acquired (sem value: {consumer_ready_sem.value}). Capturing frame...")

        # 2. Capture the frame
        frame = picam2.capture_array("main")
        if frame is None:
            print("Camera Producer: Warning - Failed to capture frame.")
            # Release the consumer_ready semaphore so the consumer can signal again if it was waiting
            consumer_ready_sem.release()
            time.sleep(0.1)
            continue

        # 3. Copy frame data into shared memory
        try:
            np.copyto(shared_array, frame)
        except Exception as e:
             print(f"Camera Producer: Error copying frame to shared memory: {e}")
             # Release consumer sem before exiting on error
             consumer_ready_sem.release()
             running = False
             break

        # 4. Signal the consumer that a new frame is ready (release frame_ready_sem)
        # This increments the semaphore value, unblocking the consumer's acquire.
        print(f"Camera Producer: Frame copied. Signaling frame ready (sem value before release: {frame_ready_sem.value})...")
        frame_ready_sem.release()
        print(f"Camera Producer: Frame ready signal sent (sem value after release: {frame_ready_sem.value}).")


except Exception as e:
    print(f"Camera Producer: An unexpected error occurred in the main block: {e}")
    import traceback
    traceback.print_exc()

finally:
    # --- Cleanup ---
    print("Camera Producer: Final cleanup...")
    if picam2 is not None:
        try: picam2.stop(); print("Camera Producer: Picamera2 stopped.")
        except Exception as e: print(f"Camera Producer: Error stopping Picamera2: {e}")

    # Close semaphores
    if frame_ready_sem:
        try: frame_ready_sem.close(); print("Camera Producer: frame_ready_sem closed.")
        except Exception as e: print(f"Error closing frame_ready_sem: {e}")
    if consumer_ready_sem:
        try: consumer_ready_sem.close(); print("Camera Producer: consumer_ready_sem closed.")
        except Exception as e: print(f"Error closing consumer_ready_sem: {e}")

    # Unlink semaphores (only do this in one process, e.g., producer)
    try:
        posix_ipc.unlink_semaphore(FRAME_READY_SEM_NAME)
        print(f"Camera Producer: Unlinked semaphore '{FRAME_READY_SEM_NAME}'.")
    except posix_ipc.ExistentialError: pass # Already unlinked
    except Exception as e: print(f"Error unlinking frame_ready_sem: {e}")
    try:
        posix_ipc.unlink_semaphore(CONSUMER_READY_SEM_NAME)
        print(f"Camera Producer: Unlinked semaphore '{CONSUMER_READY_SEM_NAME}'.")
    except posix_ipc.ExistentialError: pass # Already unlinked
    except Exception as e: print(f"Error unlinking consumer_ready_sem: {e}")


    # Close and unlink shared memory
    if shm is not None:
        try:
            shm.close()
            print("Camera Producer: Shared memory closed.")
            shm.unlink() # Unlink only in producer
            print("Camera Producer: Shared memory unlinked.")
        except FileNotFoundError: pass # Already unlinked
        except Exception as e: print(f"Camera Producer: Error during shared memory cleanup: {e}")

    print("Camera Producer: Exiting.")
