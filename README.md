# traktor2resolume

This python script enables synchronization between Traktor and Resolume using
timecode data. It allows for real-time audio-visual synchronization, even when
Traktor and Resolume are running on separate computers.

## How it works

- Screenshots a user-defined area within Traktor's interface to capture relevant
  timecode information.
- Uses Tesseract for OCR (Optical Character Recognition) to extract timing data
  from the screenshots.
- Transmits this timing information to a separate process or computer using ZeroMQ (zmq).
- The script plays back two separate timecode files - one for each deck - synchronized
  according to the received timing data. These timecode files should be played on a
  virtual audio device, which can be recognized by Resolume as an SMPTE input source.

> **Important**: Due to technical limitations, it is not possible to adjust the speed of
  the timecode. Therefore, if you alter the speed of a track in Traktor, the synchronization
  will not work correctly.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Ensure your Traktor setup has the time set to ascending (0 to N), not descending (N
  to 0).
- If using separate computers, ensure they are connected to the same WiFi network, and
  the necessary ports are open.
- Define the screenshot area within Traktor's interface that contains the
  necessary timecode data.
- Install needed packages:
  - ubuntu: `sudo apt install tesseract-ocr portaudio19-dev`
  - fedora: `sudo dnf install tesseract portaudio-devel`
- Setup a virtual audio device (for instance, using
  [VoiceMeeter Banana](https://vb-audio.com/Voicemeeter/banana.htm)). and assign this
  device as the SMPTE input source within Resolume.
- Generate a timecode [here](https://elteesee.pehrhovey.net), place it at the root
  directory of the repository, and rename this file to `timecode.wav`.

## Installation

```
conda create -n timecode python=3.10
conda activate timecode
pip install -r requirements.txt
```

## Usage

Open three terminals:

1. start left desk receiver
  ```
  python timecode.py --left-desk-receiver
  ```

2. start right desk receiver
  ```
  python timecode.py --right-desk-receiver
  ```

3. start sender
  ```
  python timecode.py --sender --left-desk-area=X1,Y1,X2,Y2  --right-desk-area=X1,Y1,X2,Y2
  ```

  for example (with `traktor.jpg`):
  ```
  python timecode.py --sender --left-desk-area=768,260,860,300  --right-desk-area=2166,260,2260,300
  ```
