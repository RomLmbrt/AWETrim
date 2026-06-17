# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from .system import SystemModel, State, create_system_model_from_yaml
from .timeseries import Cycle
from .utils import defaults, reference_frames, color_palette, utils
import logging

# Configure logging (usually done once in your main module or __init__.py)
logging.basicConfig(level=logging.INFO)  # or DEBUG for more detail
logger = logging.getLogger(__name__)  # Use module-specific logger
