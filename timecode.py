import argparse
import logging
import multiprocessing
import threading
import time

# import cv2  # for testing
import numpy as np
import pytesseract
import zmq
from PIL import ImageGrab
from rich.logging import RichHandler

try:
    import pyaudio
    from pydub import AudioSegment
except ImportError:
    print("Pyaudio and/or pydub not available, do not use --<SIDE>-receiver option")
    pass


log = logging.getLogger("timecode")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true")

    # sender arguments
    parser.add_argument("--sender", action="store_true")
    parser.add_argument("--left-desk-area", type=str)
    parser.add_argument("--right-desk-area", type=str)
    parser.add_argument("--receiver-host", type=str, default="127.0.0.1")

    # receivers arguments
    parser.add_argument("--left-desk-receiver", action="store_true")
    parser.add_argument("--left-desk-receiver-port", type=int, default=30303)
    parser.add_argument("--left-audio-device-index", type=int, default=None)
    parser.add_argument("--right-desk-receiver", action="store_true")
    parser.add_argument("--right-desk-receiver-port", type=int, default=30304)
    parser.add_argument("--right-audio-device-index", type=int, default=None)

    return parser.parse_args()


def extract_screenshots(
    left_area: tuple[int, int, int, int],
    right_area: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts screenshots of specific regions from the screen."""

    screenshot = np.array(ImageGrab.grab())
    # screenshot = cv2.imread("assets/traktor.jpg")  # for testing

    x1, y1 = left_area[0], left_area[1]
    x2, y2 = left_area[2], left_area[3]
    left_screenshot = screenshot[y1:y2, x1:x2]

    x1, y1 = right_area[0], right_area[1]
    x2, y2 = right_area[2], right_area[3]
    right_screenshot = screenshot[y1:y2, x1:x2]

    return left_screenshot, right_screenshot


def serve_screenshots(left_area, left_queue, right_area, right_queue):
    log.info("<sender> start screenshot extractor")
    while True:
        left_screenshot, right_screenshot = extract_screenshots(left_area, right_area)
        left_queue.put(left_screenshot)
        right_queue.put(right_screenshot)
        time.sleep(1)


def extract_timecode(screenshot: np.ndarray) -> int:
    """Extracts a timecode from a screenshot image."""

    text = pytesseract.image_to_string(
        screenshot,
        config="-c tessedit_char_whitelist='1234567890:.-'",
    )

    # convert to msecs
    timecode = [int(t) for t in text.strip().split(":")]
    timecode = timecode[0] * 60 + timecode[1]
    timecode = timecode * 1000

    return timecode


def serve_timecodes(host: str, port: int, queue: multiprocessing.Queue, side: str):
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    url = f"tcp://{host}:{port}"
    sender.bind(url)

    log.info(f"<sender> start timecode extractor (push to '{url}')")

    # get the intial timecode

    while True:
        try:
            screenshot = queue.get()
            last_timecode = extract_timecode(screenshot)
            log.debug(f"<{side}-sender> extracted timecode {last_timecode}")
            break
        except (SystemError, ValueError, IndexError):
            log.error(
                f"<sender ({side})> ERROR impossible to extract "
                + "timecode from screenshot"
            )
            time.sleep(1)

    while True:
        screenshot = queue.get()

        try:
            start = time.time()
            new_timecode = extract_timecode(screenshot)

            if new_timecode == last_timecode:
                log.debug(f"<sender-{side}> no change")
                continue

            sender.send_string(str(new_timecode))
            last_timecode = new_timecode

            log.debug(
                f"<sender-{side}> data sent (timecode={new_timecode} "
                + f" loop_duration={time.time() - start:.3f} secs, to={url})"
            )
        except (SystemError, ValueError, IndexError) as error:
            log.error(
                f"<sender ({side})> ERROR impossible to extract "
                + f"timecode from screenshot: {error}"
            )


def sender(
    host: str,
    left_port: int,
    left_area: tuple[int, ...],
    right_port: int,
    right_area: tuple[int, ...],
):
    left_queue: multiprocessing.Queue = multiprocessing.Queue()
    left_process = multiprocessing.Process(
        target=serve_timecodes,
        args=(host, left_port, left_queue, "left"),
    )
    left_process.start()

    right_queue: multiprocessing.Queue = multiprocessing.Queue()
    right_process = multiprocessing.Process(
        target=serve_timecodes,
        args=(host, right_port, right_queue, "right"),
    )
    right_process.start()

    screenshot_process = multiprocessing.Process(
        target=serve_screenshots,
        args=(left_area, left_queue, right_area, right_queue),
    )
    screenshot_process.start()

    left_process.join()
    right_process.join()
    screenshot_process.join()


def receiver(
    device_index: int,
    port: int,
    position: str,
):
    # define audio output device for each board

    p = pyaudio.PyAudio()

    if device_index is None:
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get("deviceCount")
        print("Available devices:")
        for i in range(num_devices):
            print(f"[{i}]", p.get_device_info_by_index(i).get("name"))
        device_index = int(input("Chose a device index: "))

    audio = AudioSegment.from_file("timecode.wav")

    stream = p.open(
        format=p.get_format_from_width(audio.sample_width),
        channels=audio.channels,
        rate=audio.frame_rate,
        output=True,
        output_device_index=device_index,
    )

    stop_signal = threading.Event()

    def play_audio(audio_segment):
        chunk_size = 100
        for chunk in audio_segment[::chunk_size]:
            if stop_signal.is_set():
                break
            stream.write(chunk.raw_data)

    # launch process for handling each board

    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    url = f"tcp://0.0.0.0:{port}"
    receiver.connect(url)

    t = threading.Thread(
        target=play_audio,
        args=(audio,),
    )
    t.start()
    log.info(f"<{position}-receiver> start modulating timecode (at '{url}')")

    timecode, start, change = 1, time.time(), False
    while True:
        new_audio = audio

        log.debug(f"<{position}-receiver> wait for a timecode...")
        new_timecode = int(receiver.recv_string())

        elapsed_time = (time.time() - start) * 1000
        start = time.time()

        # update internal timecode
        timecode = int(timecode + elapsed_time)

        log.info(
            f"<{position}-receiver> New data received: (new_timecode={new_timecode}, "
            + f"old_timecode={timecode}, diff={new_timecode / timecode:.2f})"
        )

        if new_timecode >= timecode * 1.05 or new_timecode <= timecode * 0.99:
            change = True
            log.debug(f"<{position}-receiver> change timecode")

        if not change:
            log.debug(f"<{position}-receiver> no change")
            continue
        timecode = new_timecode

        new_audio = new_audio[new_timecode:]

        stop_signal.set()
        t.join()
        stop_signal.clear()

        t = threading.Thread(
            target=play_audio,
            args=(new_audio,),
        )
        t.start()

        change = False
        start = time.time()


if __name__ == "__main__":
    args = get_args()

    handler = RichHandler(
        show_path=False,
        omit_repeated_times=False,
        log_time_format="[%H:%M:%S]",
        markup=True,
    )

    for package in ["pydub"]:
        logging.getLogger(package).propagate = False

    logging.basicConfig(
        format="%(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
        handlers=[handler],
    )

    if args.left_desk_receiver and args.right_desk_receiver:
        raise ValueError("--left-desk-receiver OR --right-desk-receiver, not both")

    if args.sender:
        areas = []
        for area in [args.left_desk_area, args.right_desk_area]:
            splitted_area = area.split(",")
            if len(splitted_area) != 4:
                raise ValueError(
                    "Invalid format for desk area coordinates. Please provide "
                    + "the coordinates in the format 'X1,Y1,X2,Y2', with commas "
                    + f"separating the values. Received value: '{area}'. "
                    + "Please check your input and try again."
                )
            areas.append(tuple([int(v) for v in splitted_area]))

        sender(
            args.receiver_host,
            args.left_desk_receiver_port,
            areas[0],
            args.right_desk_receiver_port,
            areas[1],
        )

    elif args.left_desk_receiver:
        receiver(args.left_audio_device_index, args.left_desk_receiver_port, "left")

    elif args.right_desk_receiver:
        receiver(args.right_audio_device_index, args.right_desk_receiver_port, "right")

    else:
        raise ValueError(
            "Missing required option. Please include '--sender', "
            + "'--left-desk-receiver', or '--right-desk-receiver'. Refer to the "
            + "documentation for further guidance."
        )
