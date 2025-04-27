## Description

This repository contains the code for a senior design/research project developed at Arkansas Tech University. The project aims to implement real-time semantic segmentation on a Raspberry Pi using Coral acceleration.

It utilizes the YOLOv11n model via the `ultralytics` library and leverages Google's `pycoral` library for hardware acceleration on Coral devices.

**Note:** This project was developed for academic purposes (undergraduate senior design/research).

## Features

* [List key features, e.g., Real-time object detection]
* [Uses YOLOv11n model]
* [Accelerated inference using Google Coral Edge TPU]
* [Built with Python and libraries like NumPy, Picamera2]

## Installation

```bash
# 1. Clone the repository
git clone [Your Repository URL]
cd [Your Repository Directory]

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt

# 4. Follow setup instructions for pycoral (if necessary)
# See: [https://coral.ai/software/#pycoral-api](https://coral.ai/software/#pycoral-api)

# 5. Follow setup instructions for picamera2 (if necessary)
# See: [https://github.com/raspberrypi/picamera2](https://github.com/raspberrypi/picamera2)
(Add any other specific setup steps required)Usage# Example command to run the main script
python main.py [arguments]
(Provide clear instructions on how to run your code)LicensingThis project combines code developed by the authors with several third-party libraries under various licenses.Project CodeThe original code specific to this project (i.e., the code written by the project authors, excluding third-party libraries) is licensed under the MIT License. A copy of the MIT License can be found in the LICENSE file in the root of this repository. You are free to use, modify, and distribute this specific code under the terms of the MIT License.MIT License

Copyright (c) [Year] [Your Name/Group Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
(Consider adding a separate LICENSE file with the full MIT text and referencing it here)DependenciesThis project relies on several third-party libraries with their own licenses. Compliance with these licenses is required when using or distributing this project. Key dependencies include:ultralytics (YOLOv11n Model): Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).Implication: This is a strong copyleft license. If you distribute this project, or a modified version of it, the AGPL-3.0 requires that the entire combined work be licensed under AGPL-3.0 and that the complete corresponding source code be made available. If you run a modified version on a network server and allow users to interact with it, you must also provide the source code to those users under the AGPL-3.0.License details: https://github.com/ultralytics/ultralytics/blob/main/LICENSEpycoral: Licensed under the Apache License 2.0.This is a permissive license allowing use and modification, but requires preservation of copyright and license notices.License details: https://github.com/google-coral/pycoral/blob/master/LICENSEPython: The project is written in Python (versions 3.9/3.11 used during development). Python is distributed under the Python Software Foundation License (PSFL), a permissive, GPL-compatible license.License details: https://docs.python.org/3/license.htmlNumPy: Licensed under the BSD 3-Clause "New" or "Revised" License.License details: https://github.com/numpy/numpy/blob/main/LICENSE.txtposix_ipc: Typically licensed under the MIT License. (Verify the specific version you installed if necessary).License details: (Usually found in the package distribution or source repository, e.g., https://github.com/osvenskan/posix_ipc/blob/master/LICENSE)picamera2: Licensed under the BSD 3-Clause "New" or "Revised" License.License details: https://github.com/raspberrypi/picamera2/blob/main/LICENSESummary: While the original contributions to this project are under the permissive MIT license, the inclusion of the ultralytics library means that distribution of the combined work is governed by the terms of the AGPL-3.0 license. Please ensure you understand and comply with the terms of all included licenses, particularly the AGPL-3.0, if you plan to modify, distribute, or deploy this project.AcknowledgementsThis project utilizes the powerful YOLO models and library provided by Ultralytics (https://ultralytics.com/).Hardware acceleration is made possible by Google's Coral platform and the pycoral library (https://coral.ai/).Built upon the extensive Python ecosystem, including libraries like NumPy and Picamera2.[Acknowledge supervisors, funding sources, etc., if applicable]Contact
