# inference_consumer.py
# Reads frames from shared memory, performs inference, and displays results.

#==========<NECESSARY PACKAGES>================
# desc:  python3.9 / pycoral (and necessary runtimes) / numpy / opencv-python / posix_ipc
#
#==========<ACTIVATE_2nd_VIRTUAL_ENVIRONMENT_IN_NEW_TERMINAL>===================
# $ source my_coral_env/bin/activate
#
#==========RUN_THIS_SCRIPT_2ND=================
# python /home/TrashBot/yolo_IPC/inference_consumer.py --model=/home/TrashBot/custom_yolo/best_full_integer_quant_edgetpu.tflite --conf_thres=0.6


import os
import sys
import argparse
import time
import cv2 # OpenCV for NMS, drawing, resizing
import numpy as np
from multiprocessing import shared_memory
import posix_ipc # Use POSIX IPC for semaphores
import signal # To handle termination gracefully
import errno

# Import PyCoral and TFLite Runtime components
try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
except ImportError:
    print("ERROR: PyCoral libraries not found. Make sure you are running this in the correct Python 3.9 environment.")
    sys.exit(1)

# --- Configuration (MUST MATCH PRODUCER) ---
FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS = 640, 480, 3 # Picamera resolution
FRAME_DTYPE = np.uint8
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS * np.dtype(FRAME_DTYPE).itemsize
SHM_NAME = 'psm_frame_buffer' # Shared memory block name
FRAME_READY_SEM_NAME = '/psm_frame_ready_sem' # Flag: Producer -> Consumer
CONSUMER_READY_SEM_NAME = '/psm_consumer_ready_sem' # Flag: Consumer -> Producer
# --- End Configuration ---


# === Constants ===
MODEL_INPUT_SIZE = (224, 224) # Fixed input size for the Edge TPU model (H, W)
DEFAULT_CONFIDENCE_THRESHOLD = 0.45 # Default Detection confidence threshold
DEFAULT_IOU_THRESHOLD = 0.45      # Default NMS IoU threshold
DEFAULT_MASK_THRESHOLD = 0.5       # Default Mask probability threshold
FPS_AVERAGE_LENGTH = 50            # Average FPS over fewer frames
MASK_ALPHA = 0.4                   # Transparency for mask overlay
WAITKEY_DELAY_MS = 5               # Delay for cv2.waitKey()

# --- Visualization Settings ---
BOX_LINE_WIDTH = 2
OPENCV_FONT = cv2.FONT_HERSHEY_SIMPLEX
OPENCV_FONT_SCALE = 0.5
OPENCV_FONT_THICKNESS = 1

# Using print for logging
def print_info(msg): print(f"INFO: {msg}")
def print_debug(msg): print(f"DEBUG: {msg}")
def print_error(msg): print(f"ERROR: {msg}")


# === Helper Functions (From original script) ===

def sigmoid(x):
    # Use np.clip to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1. / (1. + np.exp(-x))

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_y = (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[..., [0, 2]] -= pad_x
    coords[..., [1, 3]] -= pad_y
    coords[..., :4] /= gain
    coords[..., [0, 2]] = coords[..., [0, 2]].clip(0, img0_shape[1])
    coords[..., [1, 3]] = coords[..., [1, 3]].clip(0, img0_shape[0])
    return coords

def crop_mask(mask, box):
    h, w = mask.shape
    x1, y1, x2, y2 = map(int, box)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    cropped_mask = np.zeros_like(mask, dtype=np.uint8)
    if x1 < x2 and y1 < y2:
      cropped_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return cropped_mask

def process_masks(protos, masks_in, boxes_model_coords, model_input_shape, orig_shape, mask_threshold=0.5):
    proto_h, proto_w, num_protos = protos.shape
    num_dets = masks_in.shape[0]
    final_masks = []
    if num_dets == 0: return final_masks
    try:
        protos_flat = protos.reshape(proto_h * proto_w, num_protos).T
        mask_activations_flat = np.matmul(masks_in, protos_flat)
        mask_activations = mask_activations_flat.reshape(num_dets, proto_h, proto_w)
        mask_probs = sigmoid(mask_activations) # Apply sigmoid to mask activations

        scale_h = proto_h / model_input_shape[0]
        scale_w = proto_w / model_input_shape[1]
        boxes_scaled_to_proto = boxes_model_coords.copy()
        boxes_scaled_to_proto[:, [0, 2]] *= scale_w
        boxes_scaled_to_proto[:, [1, 3]] *= scale_h

        for i in range(num_dets):
            binary_mask_proto_res = (mask_probs[i] > mask_threshold).astype(np.uint8)
            cropped_binary_mask = crop_mask(binary_mask_proto_res, boxes_scaled_to_proto[i])
            # Use INTER_NEAREST for binary masks to avoid introducing intermediate values
            upsampled_mask = cv2.resize(
                cropped_binary_mask,
                (orig_shape[1], orig_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            final_masks.append(upsampled_mask)
    except Exception as e:
        print_error(f"Error in process_masks: {e}")
        # Return empty list on error
        return []

    return final_masks

# --- Global variables for cleanup ---
shm = None
running = True # Flag to control the main loop
frame_ready_sem = None
consumer_ready_sem = None

def cleanup(signum, frame):
    """Signal handler for graceful cleanup."""
    global running, shm
    print_info("Inference Consumer: Signal received, initiating cleanup...")
    running = False # Stop the main loop

signal.signal(signal.SIGINT, cleanup) # Handle Ctrl+C
signal.signal(signal.SIGTERM, cleanup) # Handle termination signals


# === Main Execution ===
print_info("Inference Consumer: Starting...")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Edge TPU YOLO-Seg Inference Consumer")
parser.add_argument('--model', help='Path to Edge TPU compiled TFLite model file (.tflite)', required=True)
parser.add_argument('--conf_thres', type=float, help='Object confidence threshold', default=DEFAULT_CONFIDENCE_THRESHOLD)
parser.add_argument('--iou_thres', type=float, help='IOU threshold for Non-Max Suppression', default=DEFAULT_IOU_THRESHOLD)
parser.add_argument('--mask_thres', type=float, help='Mask probability threshold', default=DEFAULT_MASK_THRESHOLD)
args = parser.parse_args()

model_path = args.model
conf_threshold = args.conf_thres
iou_threshold = args.iou_thres
mask_threshold = args.mask_thres

if not os.path.exists(model_path) or not model_path.endswith('.tflite'):
    print_error(f'Model path {model_path} is invalid or not a .tflite file.')
    sys.exit(1)

# --- Attach to Shared Memory and Semaphores ---
print_info("Inference Consumer: Attempting to attach to shared resources...")
try:
    # Wait a moment for producer to potentially create them
    time.sleep(2)
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    frame_ready_sem = posix_ipc.Semaphore(FRAME_READY_SEM_NAME)
    print_info(f"Inference Consumer: Attached to semaphore '{FRAME_READY_SEM_NAME}'")
    consumer_ready_sem = posix_ipc.Semaphore(CONSUMER_READY_SEM_NAME)
    print_info(f"Inference Consumer: Attached to semaphore '{CONSUMER_READY_SEM_NAME}'")
    shared_array = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=FRAME_DTYPE, buffer=shm.buf)
    print_info("Inference Consumer: Attached to shared memory.")
except FileNotFoundError:
     print_error(f"Inference Consumer: Shared resources not found. Is the producer running and did it create them?")
     sys.exit(1)
except Exception as e:
    print_error(f"Inference Consumer: Error attaching to shared resources: {e}")
    sys.exit(1)


# --- Load Edge TPU Model ---
print_info(f"Inference Consumer: Loading Edge TPU model from: {model_path}")
try:
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    print_info("Inference Consumer: Interpreter loaded successfully.")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_height, input_width = common.input_size(interpreter)
    if (input_height, input_width) != MODEL_INPUT_SIZE:
         print_error(f"Model expects input size {input_height}x{input_width}, script uses {MODEL_INPUT_SIZE}.")
         sys.exit(1)
    input_dtype = input_details[0]['dtype']
    output_quant_params = {}
    print_info("--- Output Tensor Quantization Details ---")
    for i, details in enumerate(output_details):
        try:
            quant_params = details['quantization']
            if isinstance(quant_params, (list, tuple)) and len(quant_params) >= 2:
                 scale, zero_point = quant_params[0], quant_params[1]
            elif isinstance(quant_params, dict):
                 scale = quant_params.get('scale', 0.0)
                 zero_point = quant_params.get('zero_point', 0)
            else:
                 scale, zero_point = 0.0, 0
                 print_error(f"  Output {i} ({details.get('name', 'N/A')}): Unexpected quantization format: {quant_params}")
            output_quant_params[i] = {'scale': scale, 'zero_point': zero_point}
            print_info(f"  Output {i} ({details.get('name', 'N/A')}): scale={scale}, zero_point={zero_point}")
        except Exception as qe:
            print_error(f"Error getting quantization params for output {i}: {qe}")
            output_quant_params[i] = {'scale': 0.0, 'zero_point': 0}
    print_info("------------------------------------------")
except Exception as e:
    print_error(f"Error loading model or interpreter: {e}")
    if shm is not None: shm.close()
    sys.exit(1)


# --- Initialize Loop Variables ---
average_frame_rate = 0
frame_rate_buffer = []
frame_count = 0

print_info("Inference Consumer: Starting inference loop... Press 'q' in the display window to quit.")
# === Begin Inference Loop ===
while running:
    loop_start_time = time.perf_counter()

    try:
        # 1. Wait for the producer signal that a frame is ready (acquire frame_ready_sem)
        print_debug(f"Inference Consumer: Waiting for frame ready signal (sem value: {frame_ready_sem.value})...")
        try:
            frame_ready_sem.acquire(timeout=2.0) # Wait up to 2 seconds
        except posix_ipc.BusyError:
            if not running: break
            print_debug("Inference Consumer: Timed out waiting for frame ready signal.")
            continue
        except Exception as e:
             print_error(f"Inference Consumer: Error acquiring frame_ready_sem: {e}")
             running = False
             break

        print_debug(f"Inference Consumer: Frame ready signal acquired (sem value: {frame_ready_sem.value}). Processing...")

        # 2. Access frame data from shared memory - MAKE A COPY
        try:
            # Assume data is BGR directly from producer (or whatever OpenCV expects)
            frame_bgr = shared_array.copy()
            print_debug("Inference Consumer: Frame received (assuming BGR).")
        except Exception as e:
            print_error(f"Error copying frame from shared memory: {e}")
            try: consumer_ready_sem.release() # Try to unblock producer
            except Exception: pass
            continue

        # --- Start Processing ---
        original_shape = frame_bgr.shape[:2] # H, W

        # --- Preprocessing for Edge TPU model ---
        # Resize the frame (still BGR) for model input
        input_image_resized = cv2.resize(frame_bgr, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
        # Model might expect RGB input, convert the resized image *only* for inference
        input_image_rgb = cv2.cvtColor(input_image_resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_image_rgb, axis=0).astype(input_dtype)

        # --- Run Inference ---
        common.set_input(interpreter, input_data)
        interpreter.invoke()

        # --- Get Outputs & Dequantize (No changes needed here) ---
        output0_int8 = interpreter.get_tensor(output_details[0]['index'])
        output1_int8 = interpreter.get_tensor(output_details[1]['index'])
        params0 = output_quant_params[0]
        protos_float = (output0_int8.astype(np.float32) - params0['zero_point']) * params0['scale'] if params0['scale'] != 0 else output0_int8.astype(np.float32)
        params1 = output_quant_params[1]
        dets_coeffs_float = (output1_int8.astype(np.float32) - params1['zero_point']) * params1['scale'] if params1['scale'] != 0 else output1_int8.astype(np.float32)
        protos = protos_float[0]
        dets_coeffs = dets_coeffs_float[0].T

        # --- Decode Detections (No changes needed here) ---
        final_boxes_model_coords = []
        final_scores = []
        final_coeffs = []
        num_dets = 0
        if len(dets_coeffs) > 0:
            boxes_raw = dets_coeffs[:, 0:4]
            scores_raw = dets_coeffs[:, 4:5]
            coeffs_raw = dets_coeffs[:, 5:37]
            conf_scores = scores_raw.flatten() # No sigmoid
            conf_mask = conf_scores >= conf_threshold
            boxes_filtered = boxes_raw[conf_mask]
            scores_filtered = conf_scores[conf_mask]
            coeffs_filtered = coeffs_raw[conf_mask]
            num_boxes_before_nms = boxes_filtered.shape[0]
            print_debug(f"Boxes BEFORE NMS = {num_boxes_before_nms}")

            if num_boxes_before_nms > 0:
                boxes_xywh_abs = boxes_filtered.copy()
                boxes_xywh_abs[:, 0] *= input_width
                boxes_xywh_abs[:, 1] *= input_height
                boxes_xywh_abs[:, 2] *= input_width
                boxes_xywh_abs[:, 3] *= input_height
                boxes_xyxy_abs = xywh2xyxy(boxes_xywh_abs)
                indices = cv2.dnn.NMSBoxes(
                    boxes_xyxy_abs.tolist(), scores_filtered.tolist(),
                    score_threshold=conf_threshold, nms_threshold=iou_threshold
                )
                if indices is not None and len(indices) > 0:
                    if isinstance(indices, tuple): indices = indices[0]
                    if len(indices) > 0 and isinstance(indices[0], (list, np.ndarray)):
                       indices = indices.flatten()
                    try:
                        final_boxes_model_coords = boxes_xyxy_abs[indices]
                        final_scores = scores_filtered[indices]
                        final_coeffs = coeffs_filtered[indices]
                        num_dets = len(final_scores)
                        print_debug(f"Detections AFTER NMS = {num_dets}")
                    except IndexError as e: print_error(f"IndexError: {e}"); num_dets = 0
                    except Exception as e: print_error(f"Error indexing: {e}"); num_dets = 0
                else: num_dets = 0
            else: num_dets = 0

        # --- Generate Masks (No changes needed here) ---
        final_masks = []
        if num_dets > 0:
            final_masks = process_masks(
                protos, final_coeffs, final_boxes_model_coords,
                MODEL_INPUT_SIZE, original_shape, mask_threshold=mask_threshold
            )

        # --- Visualize ---
        # Use the original frame_bgr directly for drawing
        img_draw = frame_bgr.copy()
        overlay = img_draw.copy()
        object_count = num_dets

        if num_dets > 0 and len(final_masks) == num_dets:
            final_boxes_orig_coords = scale_coords(MODEL_INPUT_SIZE, final_boxes_model_coords.copy(), original_shape)
            np.random.seed(42)
            colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(num_dets)]
            for i in range(num_dets):
                box = final_boxes_orig_coords[i]
                score = final_scores[i]
                mask = final_masks[i]
                color = colors[i] # Color is BGR
                colored_mask = np.zeros_like(img_draw, dtype=np.uint8)
                colored_mask[mask == 1] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, MASK_ALPHA, 0)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, BOX_LINE_WIDTH)
                label = f'Common_litter ({score:.2f})'
                (lw, lh), base = cv2.getTextSize(label, OPENCV_FONT, OPENCV_FONT_SCALE, OPENCV_FONT_THICKNESS)
                ty_base = y1 - base; ty_top = ty_base - lh - 2
                bx_top = max(ty_top, 0); tx_draw_y = bx_top + lh + 1
                cv2.rectangle(img_draw, (x1, bx_top), (x1 + lw + 2, bx_top + lh + base + 2), color, -1)
                cv2.putText(img_draw, label, (x1 + 1, tx_draw_y), OPENCV_FONT, OPENCV_FONT_SCALE, (0, 0, 0), OPENCV_FONT_THICKNESS)
        elif num_dets > 0: print_error("Mask/detection count mismatch.")

        final_image_cv = cv2.addWeighted(img_draw, 1.0, overlay, MASK_ALPHA, 0) if num_dets > 0 else img_draw # Result is BGR

        # Calculate and draw FPS
        loop_stop_time = time.perf_counter()
        fps = (1 / (loop_stop_time - loop_start_time)) if (loop_stop_time > loop_start_time) else 0
        frame_rate_buffer.append(fps)
        if len(frame_rate_buffer) > FPS_AVERAGE_LENGTH: frame_rate_buffer.pop(0)
        avg_fps = np.mean(frame_rate_buffer) if frame_rate_buffer else 0
        cv2.putText(final_image_cv, f'FPS: {avg_fps:.1f}', (10, 20), OPENCV_FONT, 0.6, (0, 255, 255), 2)
        cv2.putText(final_image_cv, f'Objects: {object_count}', (10, 40), OPENCV_FONT, 0.6, (0, 255, 255), 2)
        print_debug(f'Current FPS: {fps:.1f}, Avg FPS: {avg_fps:.1f}')


        # --- Display detection results ---
        print_debug("Attempting to display detection results...")
        cv2.imshow('Coral YOLO-Seg Results', final_image_cv) # Display BGR image
        print_debug("Finished imshow call.")


        # --- Handle Keypresses ---
        print_debug(f"Waiting for key press ({WAITKEY_DELAY_MS}ms)...")
        key = cv2.waitKey(WAITKEY_DELAY_MS) & 0xFF
        print_debug(f"waitKey returned key code: {key}")
        if key == ord('q') or key == ord('Q'):
            print_info("Quit key pressed. Exiting.")
            running = False
            break

        # 3. Signal producer that we are ready for the next frame
        print_debug(f"Signalling consumer ready for next frame (sem value before release: {consumer_ready_sem.value})...")
        consumer_ready_sem.release()
        print_debug(f"Consumer ready signal sent (sem value after release: {consumer_ready_sem.value}).")


    except KeyboardInterrupt:
        print_info("Keyboard interrupt detected. Exiting.")
        running = False
    except Exception as e:
        print_error(f"An error occurred during the loop: {e}")
        import traceback
        traceback.print_exc()
        try:
            if consumer_ready_sem: consumer_ready_sem.release()
        except Exception: pass
        time.sleep(1)

# === Clean up ===
print_info(f'Average pipeline FPS: {average_frame_rate:.2f}')
print_info("Closing OpenCV windows...")
cv2.destroyAllWindows()

# Close shared memory and semaphores for this process
if frame_ready_sem:
    try: frame_ready_sem.close(); print_info("frame_ready_sem closed.")
    except Exception as e: print_error(f"Error closing frame_ready_sem: {e}")
if consumer_ready_sem:
    try: consumer_ready_sem.close(); print_info("consumer_ready_sem closed.")
    except Exception as e: print_error(f"Error closing consumer_ready_sem: {e}")
if shm is not None:
    try: shm.close(); print_info("Shared memory closed.")
    except Exception as e: print_error(f"Error closing shared memory: {e}")

print_info("Inference Consumer: Exiting.")
