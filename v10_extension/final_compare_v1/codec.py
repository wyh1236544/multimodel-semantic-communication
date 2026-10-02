"""JPEG + exact compressed UTF8 baseline; self-describing CRC objects.

Quality selection sees source byte size, NOT validation PSNR, channel truth or
decoded data. Refused image and text objects are independent; empty text and a
fixed gray image remain in all-sample metrics. No partial-JPEG repair.
"""
import io
import struct
import zlib
import numpy as np
from PIL import Image
from final_compare_v1.spec import digital_samples

TEXT = struct.Struct(">4sI")
IMAGE = struct.Struct(">4sHHI")


def seal(body):
    return body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)


def unseal(data):
    data = bytes(data)
    if len(data) < 4 or zlib.crc32(data[:-4]) & 0xffffffff != struct.unpack(">I", data[-4:])[0]:
        return None
    return data[:-4]


def encode_text(text):
    raw = text.encode("utf-8")
    if len(raw) > 65536:
        raise ValueError("Caption exceeds public 64KiB limit")
    return seal(TEXT.pack(b"TXF1", len(raw)) + zlib.compress(raw, 9))


def decode_text(data):
    body = unseal(data)
    if body is None or len(body) < TEXT.size:
        return "", False
    magic, count = TEXT.unpack(body[:TEXT.size])
    if magic != b"TXF1" or count > 65536:
        return "", False
    try:
        inflater = zlib.decompressobj()
        raw = inflater.decompress(body[TEXT.size:], 65537)
        if (not inflater.eof or inflater.unconsumed_tail or inflater.unused_data or len(raw) != count):
            return "", False
        return raw.decode("utf-8", errors="strict"), True
    except (ValueError, UnicodeDecodeError, zlib.error):
        return "", False


def rgb8(values):
    values = np.asarray(values, float)
    if values.shape != (128, 128, 3) or not np.isfinite(values).all():
        raise ValueError("Expected the SAME finite 128x128 RGB business input")
    return np.rint(np.clip(values, 0, 1)*255).astype(np.uint8)


def encode_image(values, quality, subsampling):
    buffer = io.BytesIO()
    Image.fromarray(rgb8(values)).save(buffer, format="JPEG", quality=int(quality),
        subsampling=int(subsampling), optimize=False, progressive=False)
    jpeg = buffer.getvalue()
    return seal(IMAGE.pack(b"JPF1", 128, 128, len(jpeg)) + jpeg)


def decode_image(data):
    erased = np.full((128, 128, 3), .5, np.float32)
    body = unseal(data)
    if body is None or len(body) < IMAGE.size:
        return erased, False
    magic, width, height, count = IMAGE.unpack(body[:IMAGE.size])
    if magic != b"JPF1" or (width, height) != (128, 128) or count != len(body)-IMAGE.size:
        return erased, False
    try:
        with Image.open(io.BytesIO(body[IMAGE.size:])) as image:
            if image.format != "JPEG" or image.size != (128, 128):
                return erased, False
            return np.asarray(image.convert("RGB"), np.float32)/255, True
    except (OSError, ValueError):
        return erased, False


def choose_source(values, text, optical, plan, slot_samples):
    text_object = encode_text(text)
    picked, quality, minimum = None, None, None
    w = optical["waveform"]
    for q in range(plan["jpeg_quality_min"], plan["jpeg_quality_max"]+1):
        candidate = encode_image(values, q, plan["jpeg_subsampling"])
        if minimum is None:
            minimum = candidate
        count = digital_samples([len(text_object), len(candidate)], w, plan["digital_policy"])
        if count <= slot_samples:
            picked, quality = candidate, q
    # Keep source-size/resource failure rather than increasing the common budget.
    offered = picked if picked is not None else minimum
    return dict(text=text_object, image=picked, offered_image=offered, jpeg_quality=quality,
                budget_fits=picked is not None, text_source_bytes=len(text.encode("utf-8")),
                compressed_text_wire_bytes=len(text_object), image_wire_bytes=len(offered),
                predicted_samples=digital_samples([len(text_object), len(offered)], w, plan["digital_policy"]),
                quality_selection="highest_public_quality_fitting_wire_sample_budget_no_PSNR_search")

