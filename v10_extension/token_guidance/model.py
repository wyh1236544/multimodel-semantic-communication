"""Process-local hooks; no legacy files, state-dict names or RX source-text branch changed."""
import torch
from v10.channels import ReceiverCSI
from business_link.engine import identity_seed
from token_guidance.spec import VARIANTS
from token_guidance.wire import send


class Guidance:
    def __init__(self, model):
        self.model = model
        self.variant, self.diagnostic = 'D', 'matched'
        self.sender_override = self.receiver_override = None
        self.receiver_override_ok = True
        self.memory = None
        early, late = model.base.text_to_image_fusion, model.base.decoder_text_to_image_refiner
        if early is None or late is None:
            raise ValueError('This checkpoint architecture lacks the required guidance modules')
        old_early, old_late = early.forward, late.forward

        def early_forward(image, text, mask=None):
            if self.sender_override is not None:
                text, mask = self.sender_override
            return old_early(image, text, mask)

        def late_forward(image, text, reliability=None):
            # Capture ONLY decoded, received C/T memory. Never CLIP(original text).
            self.memory = text.detach().clone()
            if not VARIANTS[self.variant][1]:
                return image, image.new_zeros(())
            if self.diagnostic == 'rx_zero':
                return image, image.new_zeros(())
            if self.receiver_override is not None:
                text = self.receiver_override
                if not self.receiver_override_ok:
                    return image, image.new_zeros(())
            return old_late(image, text, reliability=reliability)

        early.forward, late.forward = early_forward, late_forward

    def set(self, variant, diagnostic='matched', sender=None, receiver=None, receiver_ok=True):
        if variant not in VARIANTS:
            raise ValueError('Unknown variant')
        self.variant, self.diagnostic = variant, diagnostic
        self.sender_override, self.receiver_override = sender, receiver
        self.receiver_override_ok = receiver_ok
        self.model.base.v10_disable_early_fusion = not VARIANTS[variant][0]

    def trainable(self, variant):
        self.set(variant)
        # Freeze VAE, CLIP, joint encoder, old Common/adapter/adaptive branches.
        shared = ('base.bottleneck.', 'base.joint_decoder.', 'base.image_head.',
                  'base.text_head.', 'base.text_direct_adapter.', 'base.text_direct_gate_logit')
        specific = (() if not VARIANTS[variant][0] else ('base.text_to_image_fusion.',))
        specific += (() if not VARIANTS[variant][1] else ('base.decoder_text_to_image_refiner.',))
        for name, p in self.model.named_parameters():
            p.requires_grad_(name.startswith(shared+specific))
        # Deterministic short fine-tuning: eval disables dropout, NOT autograd.
        self.model.eval()
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]


class TokenFrame:
    def __init__(self, profile, snr_db, batch, device, context, **kwargs):
        if batch != 1 or kwargs.get('fixed_mode') != 0 or kwargs.get('blocks', 1) != 1:
            raise ValueError('Quick screen requires batch1, public reconstruction mode, one field')
        self.profile, self.snr, self.context = profile, float(snr_db), context
        self.optical, self.blocks, self.pilots = False, 1, 256
        self.tx_mode = self.rx_mode = self.requested_mode = torch.zeros(1, dtype=torch.long, device=device)
        self.header_ok = torch.ones(1, dtype=torch.bool, device=device)
        self.feedback_rejected = torch.zeros(1, dtype=torch.bool, device=device)
        self.transport = None
        # Independent probe precedes encoding, with the same field as the payload.
        if context['clean']:
            estimated = 40.
        else:
            from business_link.engine import field_draw, Transport
            from token_guidance.spec import SETTINGS
            gain, _ = field_draw(profile, context['field_seed'])
            self.transport = Transport(SETTINGS, gain, 10**(-self.snr/10), context['noise_seed'])
            estimated = self.transport.probe()
        self.routing_snr = torch.tensor([estimated], device=device).float().clamp(-40, 40)
        noise = torch.pow(10., -self.routing_snr[:, None]/10)
        self.receiver = ReceiverCSI(torch.ones((1, 1), dtype=torch.complex64, device=device),
                                    noise, noise/256, self.routing_snr[:, None])
        self.last_stats = {}

    def payload(self, tokens, tx_amp, rx_amp, interleave=True):
        restored, self.last_stats, self.transport = send(
            tokens[0].detach().cpu().numpy(), self.profile, self.snr,
            self.context['field_seed'], self.context['noise_seed'], self.context['clean'], self.transport)
        ok = self.last_stats['feature_object_crc_accepted']
        self.header_ok.fill_(ok)
        received = torch.as_tensor(restored[None], device=tokens.device, dtype=tokens.dtype)
        # Exact physical forward, approximate straight-through backward. NO claim
        # of differentiable Viterbi/CRC/QAM. Erasures have zero sender gradient.
        if torch.is_grad_enabled() and tokens.requires_grad and ok:
            received = received+(tokens-tokens.detach())
        return received, tokens.new_full((1, 288, 1), float(ok))


def frame_context(seed, image_id, phase, condition, epoch=0):
    return dict(clean=condition[0] == 'quantized_clean',
                field_seed=identity_seed(seed, image_id, phase, condition, epoch, 'field'),
                noise_seed=identity_seed(seed, image_id, phase, condition, epoch, 'noise'))
