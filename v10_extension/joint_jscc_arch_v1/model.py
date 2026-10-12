"""Whole-text C/I/T, AR text recovery, cross-token JSCC and public RX conditioning.

Only source_encode reads source features. receive reads observed symbols and
public receiver metadata; teacher forcing is a separate training-only method.
Common supervision reads physical C slots BEFORE any private-symbol RX mixing.
"""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from joint_jscc_arch_v1.core import (WIRE_GROUPS, DECODE_GROUPS, IMAGE_INDICES,
                                     source_groups, arm_options, chunks)


def attention_mask(groups, separate, device):
    if not separate: return None
    g = torch.tensor(groups,device=device)
    return g[:,None] != g[None,:]


def normalize_wire(x):
    # Each publicly defined group has unit mean energy. Split C halves also in
    # joint arm, to avoid cross-modal source-dependent norm side paths in control.
    bounds = ((0,16),(16,32),(32,256),(256,288))
    return torch.cat([x[:,a:b]/x[:,a:b].square().mean((1,2),keepdim=True).clamp_min(1e-8).sqrt()
                      for a,b in bounds],1)


class TokenCodec(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        self.up = nn.Linear(16,hidden)
        self.position = nn.Parameter(torch.randn(1,288,hidden)*.02)
        layer = nn.TransformerEncoderLayer(hidden,4,hidden*4,dropout=0.,
                                           activation='gelu',batch_first=True,norm_first=True)
        self.mixer = nn.TransformerEncoder(layer,layers)
        self.down = nn.Linear(hidden,16)
        nn.init.zeros_(self.down.weight); nn.init.zeros_(self.down.bias)

    def forward(self,x,separate):
        h = self.mixer(self.up(x)+self.position,
                       mask=attention_mask(WIRE_GROUPS,separate,x.device))
        return x+self.down(h)


class SpatialCodec(nn.Module):
    """Local latent-grid coding; optional public missing-position mask at RX."""
    def __init__(self,receiver=False):
        super().__init__(); self.receiver=receiver
        self.net=nn.Sequential(nn.Conv2d(17 if receiver else 16,32,3,padding=1),
                               nn.GELU(),nn.Conv2d(32,16,3,padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self,tokens,present=None):
        grid=tokens.transpose(1,2).reshape(tokens.size(0),16,16,16)
        features=grid
        if self.receiver:
            if present is None: raise ValueError('Public RX image layout required')
            features=torch.cat((grid,present.transpose(1,2).reshape(tokens.size(0),1,16,16)),1)
        return (grid+self.net(features)).flatten(2).transpose(1,2)


class ARTextDecoder(nn.Module):
    def __init__(self, old, maximum, bos, eos, pad):
        super().__init__()
        h = old.output.in_features
        self.maximum,self.bos,self.eos,self.pad = maximum,bos,eos,pad
        self.embedding = nn.Embedding(old.output.out_features,h)
        self.embedding.weight.data.copy_(old.output.weight.detach())
        self.position = nn.Parameter(torch.randn(1,maximum,h)*.02)
        self.position.data[:,:min(maximum,old.queries.size(1))].copy_(old.queries.detach()[:,:maximum])
        self.decoder = copy.deepcopy(old.decoder)
        self.output = nn.Linear(h,old.output.out_features)
        self.output.weight = self.embedding.weight
        self.output.bias.data.copy_(old.output.bias.detach())

    def features(self, prefix, memory):
        n = prefix.shape[1]
        if n > self.maximum: raise ValueError('Decoder prefix exceeds capacity')
        x = self.embedding(prefix)*math.sqrt(self.embedding.embedding_dim)+self.position[:,:n]
        causal = torch.triu(torch.ones(n,n,device=x.device,dtype=torch.bool),diagonal=1)
        return self.decoder(x,memory,tgt_mask=causal)

    def logits(self,prefix,memory):
        return self.output(self.features(prefix,memory))

    def teacher_loss(self, ids, lengths, memory):
        # CLIP uses EOS also as PAD: use explicit TRAIN target lengths, not id!=PAD.
        n = int(lengths.max())-1
        logits = self.logits(ids[:,:n],memory)
        target = ids[:,1:n+1]
        mask = torch.arange(n,device=ids.device)[None,:] < (lengths-1)[:,None]
        loss = F.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1),reduction='none').reshape_as(mask)
        return (loss*mask).sum()/mask.sum().clamp_min(1)

    @torch.no_grad()
    def generate(self, memory, maximum=None):
        cap = self.maximum if maximum is None else min(maximum,self.maximum)
        out = torch.full((memory.shape[0],1),self.bos,device=memory.device,dtype=torch.long)
        done = torch.zeros(memory.shape[0],device=memory.device,dtype=torch.bool)
        for _ in range(cap-1):
            # Exactly the same last-token prediction, without allocating a full
            # prefix x 49k-vocabulary output at every autoregressive step.
            next_id = self.output(self.features(out,memory)[:,-1]).argmax(-1)
            next_id = torch.where(done,next_id.new_full(next_id.shape,self.eos),next_id)
            out = torch.cat((out,next_id[:,None]),1)
            done |= next_id==self.eos
            if bool(done.all()): break
        return out


class ArchitectureModel(nn.Module):
    def __init__(self,base,config,rrc):
        super().__init__()
        self.config = dict(config)
        h = base.hidden_dim
        enc = base.multimodal_encoder
        self.image_projection = copy.deepcopy(base.vae_tokenizer.latent_proj)
        self.common_queries = nn.Parameter(enc.common_tokens.detach().clone())
        self.image_position = nn.Parameter(enc.visual_position.detach().clone())
        self.image_type = nn.Parameter(enc.image_type.detach().clone())
        self.text_type = nn.Parameter(enc.text_type.detach().clone())
        self.text_position = nn.Parameter(torch.randn(1,config['max_text_tokens'],h)*.02)
        self.source_transformer = copy.deepcopy(enc.joint_transformer)
        self.source_norm = copy.deepcopy(enc.output_norm)
        self.text_compressor = copy.deepcopy(enc.text_compressor)
        self.text_compressor.queries = nn.Parameter(enc.text_compressor.queries[:,:32].detach().clone())
        self.bottleneck = copy.deepcopy(base.bottleneck)
        self.old_tx = copy.deepcopy(base.channel.tx_map)
        self.old_rx = copy.deepcopy(base.channel.rx_map)
        self.old_adapter = copy.deepcopy(base.channel.adapter)
        self.modern_tx = TokenCodec(config['codec_hidden'],config['codec_layers'])
        self.modern_rx = TokenCodec(config['codec_hidden'],config['codec_layers'])
        self.spatial_tx = SpatialCodec()
        self.spatial_rx = SpatialCodec(True)
        self.rx_projection = copy.deepcopy(base.joint_decoder.input_projection)
        self.rx_transformer = copy.deepcopy(base.joint_decoder.shared_decoder)
        self.rx_norm = copy.deepcopy(base.joint_decoder.output_norm)
        self.rx_image_position = nn.Parameter(base.joint_decoder.image_position.detach().clone())
        self.rx_condition = nn.Sequential(nn.Linear(5,h),nn.GELU(),nn.Linear(h,h))
        nn.init.zeros_(self.rx_condition[-1].weight); nn.init.zeros_(self.rx_condition[-1].bias)
        self.missing_token = nn.Parameter(torch.zeros(1,1,h))
        self.image_head = copy.deepcopy(base.image_head)
        tok = base.text_encoder.tokenizer
        self.language = ARTextDecoder(base.text_head,config['max_text_tokens'],tok.bos_token_id,
                                      tok.eos_token_id,tok.pad_token_id)
        self.common_image_head = nn.Linear(h,4*4*4)
        self.register_buffer('rrc',torch.as_tensor(rrc).float().reshape(1,1,-1))
        self.options = arm_options('B1_text')
        for p in self.parameters(): p.requires_grad_(True)
        self.configure('B1_text')

    def configure(self,arm):
        self.options = arm_options(arm)
        for n,p in self.named_parameters():
            p.requires_grad_(not (n.startswith(('modern_tx.','modern_rx.','spatial_tx.','spatial_rx.','rx_condition.','missing_token'))
                                 and not self.options['modern']))
        self.eval()  # same deterministic dropout policy across all trained arms

    def source_encode(self,latent,text_features,text_mask):
        v = self.image_projection(latent).flatten(2).transpose(1,2)
        v = v+self.image_position+self.image_type
        t = text_features+self.text_position[:,:text_features.shape[1]]+self.text_type
        tokens = torch.cat((self.common_queries.expand(v.size(0),-1,-1),v,t),1)
        padding = torch.cat((torch.zeros(v.size(0),288,device=v.device,dtype=torch.bool),~text_mask.bool()),1)
        encoded = self.source_norm(self.source_transformer(tokens,
            mask=attention_mask(source_groups(t.shape[1]),self.options['separate'],v.device),
            src_key_padding_mask=padding))
        c,i = encoded[:,:32],encoded[:,32:288]
        text = self.text_compressor(encoded[:,288:],key_padding_mask=~text_mask.bool())
        z = self.bottleneck(c,i,text)
        if self.options['modern']: z['z_img']=self.spatial_tx(z['z_img'])
        indices = torch.tensor(IMAGE_INDICES,device=v.device)
        x = torch.cat((z['z_common'],z['z_img'][:,indices],z['z_txt']),1)
        x = normalize_wire(self.old_tx(normalize_wire(x)))
        if self.options['modern']: x = normalize_wire(self.modern_tx(x,self.options['separate']))
        return x

    def waveform(self,x,channel_config):
        axes = (x.reshape(1,-1,2)/math.sqrt(2)).transpose(1,2)
        wave = F.conv_transpose1d(axes.reshape(2,1,-1),self.rrc,
                                  stride=channel_config['sps']).reshape(1,2,-1)*math.sqrt(channel_config['sps'])
        power = wave.square().sum(1)
        relative = power/power.mean(1,keepdim=True).clamp_min(1e-10)
        papr = 10*torch.log10(relative.max(1).values.clamp_min(1e-10))
        penalty = F.relu(relative-10**(channel_config['papr_target_db']/10)).square().mean()
        return papr,penalty,int(wave.shape[-1])

    def receive(self,received,reliability,pilot_snr,available):
        if received.shape[1:] != (288,16): raise ValueError('Wrong wire budget')
        b = received.shape[0]
        rel = reliability.to(received).reshape(b,288,1)
        ok = available.to(received).reshape(b,1,1)
        wire = received*ok
        # These auxiliary memories have NO path from private receive symbols.
        physical_common = self.rx_projection(wire[:,:32])
        common_image = physical_common[:,:16] if self.options['separate'] else physical_common
        common_text = physical_common[:,16:] if self.options['separate'] else physical_common
        r = self.old_adapter(wire,rel)
        r = self.old_rx(r,rel)
        if self.options['modern']: r = self.modern_rx(r,self.options['separate'])
        r = r*ok
        indices = torch.tensor(IMAGE_INDICES,device=r.device)
        expanded = r.new_zeros(b,320,16)
        expanded[:,:32] = r[:,:32]
        expanded[:,32+indices] = r[:,32:256]
        expanded[:,288:] = r[:,256:]
        present = r.new_ones(b,320,1)
        present[:,32:288] = 0.; present[:,32+indices] = 1.
        rr = rel.new_zeros(b,320,1)
        rr[:,:32] = rel[:,:32]; rr[:,32+indices] = rel[:,32:256]; rr[:,288:] = rel[:,256:]
        observed = expanded.square().mean(-1,keepdim=True).clamp_min(1e-8).log().clamp(-20,20)/20
        if self.options['modern']:
            spatial=self.spatial_rx(expanded[:,32:288],present[:,32:288])
            expanded=torch.cat((expanded[:,:32],spatial,expanded[:,288:]),1)
        snr = pilot_snr.to(r).clamp(-40,40).reshape(b,1,1).expand(-1,320,-1)/40
        features = torch.cat((rr,snr,present,ok.expand(-1,320,-1),observed),-1)
        h = self.rx_projection(expanded)
        h[:,32:288] = h[:,32:288]+self.rx_image_position
        if self.options['modern']:
            h = h+self.rx_condition(features)+(1-present)*self.missing_token
        h = self.rx_norm(self.rx_transformer(h,
                mask=attention_mask(DECODE_GROUPS,self.options['separate'],h.device)))
        image_memory = torch.cat((h[:,:16],h[:,32:288]),1) if self.options['separate'] else h
        text_memory = torch.cat((h[:,16:32],h[:,288:]),1) if self.options['separate'] else h
        latent = self.image_head(image_memory,spatial_tokens=h[:,32:288])
        coarse = self.common_image_head(common_image.mean(1)).reshape(b,4,4,4)
        return dict(latent=latent,text_memory=text_memory,common_text_memory=common_text,
                    common_coarse=coarse,physical_common=wire[:,:32])


@torch.no_grad()
def encode_text(base,text,maximum,device):
    """Frozen CLIP processes <=75 content tokens per chunk; no model download."""
    tok = base.text_encoder.tokenizer
    raw = tok(text,add_special_tokens=False,truncation=False)['input_ids']
    info = chunks(raw,maximum,tok.bos_token_id,tok.eos_token_id,tok.pad_token_id)
    feature_parts = []
    first = last = None
    for ids,mask,n in info['chunks']:
        ii = torch.tensor([ids],device=device)
        mm = torch.tensor([mask],device=device)
        result = base.text_encoder.backbone(input_ids=ii,attention_mask=mm).last_hidden_state
        feature = base.text_encoder.projection(result)
        if first is None: first = feature[:,:1]
        feature_parts.append(feature[:,1:n+1]); last = feature[:,n+1:n+2]
    features = torch.cat([first]+feature_parts+[last],1)
    length = info['target_length']
    features = F.pad(features,(0,0,0,maximum-length))
    info.update(features=features.cpu(),mask=(torch.arange(maximum)<length)[None,:],
                covered_text=tok.decode(info['target_ids'][:length],skip_special_tokens=True),
                common_reference_text=tok.decode(info['target_ids'][:min(length,24)],skip_special_tokens=True))
    return info
