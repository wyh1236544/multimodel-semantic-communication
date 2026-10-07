"""Reuse CRC/JPEG/UTF8 formats, charge V3 interspersed pilots in source selection."""
from final_compare_v1.codec import encode_text, encode_image, decode_text, decode_image
from optical_realism_v3.spec import digital_samples


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

