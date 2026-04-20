from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
import streamlit as st


def show_gradcam_base64(base64_png: str, caption: str = "Grad-CAM++"):
    if not base64_png:
        st.info("No Grad-CAM available.")
        return
    data = base64.b64decode(base64_png.encode("utf-8"))
    img = Image.open(io.BytesIO(data))
    st.image(img, caption=caption, width="stretch")

