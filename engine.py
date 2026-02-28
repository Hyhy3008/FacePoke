import uuid
import logging
import os
import io
import asyncio
from async_lru import alru_cache
import base64
from typing import Dict
import numpy as np
import torch
from PIL import Image, ImageOps

from liveportrait.utils.camera import get_rotation_matrix
from liveportrait.utils.io import resize_to_limit
from liveportrait.utils.crop import prepare_paste_back, paste_back, parse_bbox_from_landmark

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_ROOT = os.environ.get('DATA_ROOT', '/tmp/data')

def base64_data_uri_to_PIL_Image(base64_string: str) -> Image.Image:
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))

class Engine:
    """
    FacePoke Engine (patched): auto lipsync temporal behavior
    """

    def __init__(self, live_portrait):
        self.live_portrait = live_portrait
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processed_cache = {}

        # NEW: per-uid animation state (for temporal lipsync)
        self.anim_state: Dict[str, Dict[str, float]] = {}

        logger.info("✅ FacePoke Engine initialized successfully.")

    @alru_cache(maxsize=512)
    async def load_image(self, data):
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')

        uid = str(uuid.uuid4())
        img_rgb = np.array(image)

        inference_cfg = self.live_portrait.live_portrait_wrapper.cfg
        img_rgb = await asyncio.to_thread(resize_to_limit, img_rgb, inference_cfg.ref_max_shape, inference_cfg.ref_shape_n)
        crop_info = await asyncio.to_thread(self.live_portrait.cropper.crop_single_image, img_rgb)
        img_crop_256x256 = crop_info['img_crop_256x256']

        I_s = await asyncio.to_thread(self.live_portrait.live_portrait_wrapper.prepare_source, img_crop_256x256)
        x_s_info = await asyncio.to_thread(self.live_portrait.live_portrait_wrapper.get_kp_info, I_s)
        f_s = await asyncio.to_thread(self.live_portrait.live_portrait_wrapper.extract_feature_3d, I_s)
        x_s = await asyncio.to_thread(self.live_portrait.live_portrait_wrapper.transform_keypoint, x_s_info)

        processed_data = {
            'img_rgb': img_rgb,
            'crop_info': crop_info,
            'x_s_info': x_s_info,
            'f_s': f_s,
            'x_s': x_s,
            'inference_cfg': inference_cfg
        }
        self.processed_cache[uid] = processed_data

        # NEW: init temporal state
        self.anim_state[uid] = {
            "phase": 0.0,
            "voice_prev": 0.0,
        }

        bbox_info = parse_bbox_from_landmark(processed_data['crop_info']['lmk_crop'], scale=1.0)

        return {
            'u': uid,
            'c': bbox_info['center'],
            's': bbox_info['size'],
            'b': bbox_info['bbox'],
            'a': bbox_info['angle'],
        }

    def _apply_auto_lipsync(self, uid: str, params: Dict[str, float]) -> Dict[str, float]:
        """
        NEW: make mouth "mấp máy nhanh khi có âm thanh", "im thì dừng/khép".
        This modifies aaa/eee/woo/smile in-place (returns the dict).
        """
        p = dict(params or {})

        # enable if user asks OR if voice is provided
        auto = float(p.get("auto_lipsync", 0.0))
        has_voice = ("voice" in p)
        if not (auto > 0.0 or has_voice):
            return p

        # read voice (0..1 or 0..25)
        voice = float(p.get("voice", 0.0))
        if voice > 1.5:
            voice_n = max(0.0, min(1.0, voice / 25.0))
        else:
            voice_n = max(0.0, min(1.0, voice))

        # if user doesn't provide voice, derive from mouth open they send
        if not has_voice:
            aaa0 = float(p.get("aaa", 0.0))
            eee0 = float(p.get("eee", 0.0))
            woo0 = float(p.get("woo", 0.0))
            voice_n = max(voice_n, max(aaa0, eee0, woo0) / 25.0)

        silence_th = float(p.get("silence_threshold", 0.06))
        hard_silence = float(p.get("hard_silence", 1.0))  # 1 = ép về 0 ngay

        # talk params
        strength = float(p.get("talk_strength", 0.85))     # 0..1
        strength = max(0.0, min(1.0, strength))

        rate_base = float(p.get("talk_rate_base", 3.8))    # Hz
        rate_gain = float(p.get("talk_rate_gain", 18.0))   # how much rate follows |Δvoice|
        rate_min = float(p.get("talk_rate_min", 2.0))
        rate_max = float(p.get("talk_rate_max", 8.0))

        # dt (fallback): assume fixed fps
        fps = float(p.get("fps", 25.0))
        fps = 25.0 if fps <= 1 else fps
        dt = 1.0 / fps

        st = self.anim_state.get(uid)
        if st is None:
            st = {"phase": 0.0, "voice_prev": 0.0}
            self.anim_state[uid] = st

        # silence: force stop
        if voice_n <= silence_th:
            if hard_silence >= 1.0:
                p["aaa"] = 0.0
                p["eee"] = 0.0
                p["woo"] = 0.0
                # smile optional: giảm về 0 để miệng không “cười nói” lúc im
                p["smile"] = float(p.get("smile", 0.0)) * 0.0
            # freeze phase so it doesn't continue "talking"
            st["voice_prev"] = voice_n
            return p

        # speaking: estimate "speech speed" from how quickly voice changes
        dv = abs(voice_n - float(st.get("voice_prev", 0.0)))
        # dv is per-frame; convert to per-second-ish by *fps
        rate = rate_base + (dv * fps) * (rate_gain / 10.0)
        rate = max(rate_min, min(rate_max, rate))

        # advance phase
        phase = float(st.get("phase", 0.0))
        phase += (2.0 * np.pi) * rate * dt
        # keep bounded
        if phase > 1e9:
            phase = phase % (2.0 * np.pi)
        st["phase"] = phase
        st["voice_prev"] = voice_n

        # base open: use provided aaa as base if present, else from voice
        base_open = float(p.get("aaa", voice_n * 22.0))
        base_open = max(0.0, min(25.0, base_open))

        # generate fast viseme cycling (aaa/eee/woo)
        s = 0.5 + 0.5 * np.sin(phase)              # 0..1
        r = 0.5 + 0.5 * np.sin(phase * 0.55 + 1.0) # slower round

        gen_aaa = base_open * (0.55 + 0.25 * (1.0 - s))     # open
        gen_eee = base_open * (0.10 + 0.55 * s)             # wide
        gen_woo = base_open * (0.05 + 0.22 * r)             # round

        # Blend with existing params (if user provides eee/woo)
        def lerp(a, b, t):
            return a * (1.0 - t) + b * t

        aaa0 = float(p.get("aaa", 0.0))
        eee0 = float(p.get("eee", 0.0))
        woo0 = float(p.get("woo", 0.0))

        tmix = strength  # could also be strength * voice_n; keep simple
        p["aaa"] = float(lerp(aaa0, gen_aaa, tmix))
        p["eee"] = float(lerp(eee0, gen_eee, tmix))
        p["woo"] = float(lerp(woo0, gen_woo, tmix))

        # Smile a bit while speaking to get "nhoẻn" feel (optional)
        smile0 = float(p.get("smile", 0.0))
        smile_gain = float(p.get("talk_smile_gain", 0.25))  # 0..1
        p["smile"] = float(np.clip(smile0 + smile_gain * voice_n, 0.0, 1.0))

        return p

    async def transform_image(self, uid: str, params: Dict[str, float]) -> bytes:
        if uid not in self.processed_cache:
            raise ValueError("cache miss")

        processed_data = self.processed_cache[uid]

        try:
            # NEW: auto lipsync behavior (only affects mouth)
            params = self._apply_auto_lipsync(uid, params)

            x_d_new = processed_data['x_s_info']['kp'].clone()

            modifications = [
                ('smile', [
                    (0, 20, 1, -0.01), (0, 14, 1, -0.02), (0, 17, 1, 0.0065), (0, 17, 2, 0.003),
                    (0, 13, 1, -0.00275), (0, 16, 1, -0.00275), (0, 3, 1, -0.0035), (0, 7, 1, -0.0035)
                ]),
                ('aaa', [
                    (0, 19, 1, 0.001), (0, 19, 2, 0.0001), (0, 17, 1, -0.0001)
                ]),
                ('eee', [
                    (0, 20, 2, -0.001), (0, 20, 1, -0.001), (0, 14, 1, -0.001)
                ]),
                ('woo', [
                    (0, 14, 1, 0.001), (0, 3, 1, -0.0005), (0, 7, 1, -0.0005), (0, 17, 2, -0.0005)
                ]),
                ('wink', [
                    (0, 11, 1, 0.001), (0, 13, 1, -0.0003), (0, 17, 0, 0.0003),
                    (0, 17, 1, 0.0003), (0, 3, 1, -0.0003)
                ]),
                ('pupil_x', [
                    (0, 11, 0, 0.0007 if params.get('pupil_x', 0) > 0 else 0.001),
                    (0, 15, 0, 0.001 if params.get('pupil_x', 0) > 0 else 0.0007)
                ]),
                ('pupil_y', [
                    (0, 11, 1, -0.001), (0, 15, 1, -0.001)
                ]),
                ('eyes', [
                    (0, 11, 1, -0.001), (0, 13, 1, 0.0003), (0, 15, 1, -0.001), (0, 16, 1, 0.0003),
                    (0, 1, 1, -0.00025), (0, 2, 1, 0.00025)
                ]),
                ('eyebrow', [
                    (0, 1, 1, 0.001 if params.get('eyebrow', 0) > 0 else 0.0003),
                    (0, 2, 1, -0.001 if params.get('eyebrow', 0) > 0 else -0.0003),
                    (0, 1, 0, -0.001 if params.get('eyebrow', 0) <= 0 else 0),
                    (0, 2, 0, 0.001 if params.get('eyebrow', 0) <= 0 else 0)
                ]),
            ]

            for param_name, adjustments in modifications:
                param_value = float(params.get(param_name, 0.0) or 0.0)
                for i, j, k, factor in adjustments:
                    x_d_new[i, j, k] += param_value * factor

            # pupil_y affects eyes
            x_d_new[0, 11, 1] -= float(params.get('pupil_y', 0.0)) * 0.001
            x_d_new[0, 15, 1] -= float(params.get('pupil_y', 0.0)) * 0.001
            params['eyes'] = float(params.get('eyes', 0.0)) - float(params.get('pupil_y', 0.0)) / 2.0

            # rotation
            R_new = get_rotation_matrix(
                processed_data['x_s_info']['pitch'] + float(params.get('rotate_pitch', 0.0)),
                processed_data['x_s_info']['yaw'] + float(params.get('rotate_yaw', 0.0)),
                processed_data['x_s_info']['roll'] + float(params.get('rotate_roll', 0.0))
            )
            x_d_new = processed_data['x_s_info']['scale'] * (x_d_new @ R_new) + processed_data['x_s_info']['t']

            # stitching + decode
            x_d_new = await asyncio.to_thread(self.live_portrait.live_portrait_wrapper.stitching, processed_data['x_s'], x_d_new)
            out = await asyncio.to_thread(self.live_portrait.live_portrait_wrapper.warp_decode, processed_data['f_s'], processed_data['x_s'], x_d_new)
            I_p = await asyncio.to_thread(self.live_portrait.live_portrait_wrapper.parse_output, out['out'])

            # paste back
            mask_ori = await asyncio.to_thread(
                prepare_paste_back,
                processed_data['inference_cfg'].mask_crop, processed_data['crop_info']['M_c2o'],
                dsize=(processed_data['img_rgb'].shape[1], processed_data['img_rgb'].shape[0])
            )
            I_p_to_ori_blend = await asyncio.to_thread(
                paste_back,
                I_p[0], processed_data['crop_info']['M_c2o'], processed_data['img_rgb'], mask_ori
            )
            result_image = Image.fromarray(I_p_to_ori_blend)

            buffered = io.BytesIO()
            result_image.save(buffered, format="WebP", quality=82, lossless=False, method=6)
            return buffered.getvalue()

        except Exception as e:
            raise ValueError(f"Failed to modify image: {str(e)}")
