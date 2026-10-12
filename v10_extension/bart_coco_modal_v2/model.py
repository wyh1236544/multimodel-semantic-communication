"""Same registered BART-V1 layers; public routing and source-isolated modes."""
import torch
from bart_clip_joint_v1.model import JointBridge,unit
from bart_coco_modal_v2.core import indices


class ModalBridge(JointBridge):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.modality='joint'; self.private_diagnostic=False

    def configure_mode(self,mode,private_diagnostic=False):
        indices(mode)
        if private_diagnostic and mode!='text_only': raise ValueError('Private diagnostic is text-only')
        self.modality,self.private_diagnostic=mode,private_diagnostic; self.arm='J1_joint'
        image=('image_tx.','image_rx.','image_query.','image_cross.','image_residual.','image_common.','clip_image.')
        text=('text_fold.','text_source.','text_source_position','text_tx.','text_rx.','text_receiver.',
              'text_receiver_position','text_expand.','bart_memory_norm.','bart_memory_adapter.','text_cross.','text_gate.','bart_common.','clip_text.')
        common=('common_queries','common_attention.','common_norm.','common_tx.','common_rx.','common_alignment.')
        allowed=(image if mode=='image_only' else text if mode=='text_only' else image+text)+common
        for n,p in self.named_parameters():
            active=not n.startswith(('bart.','image_reference.')) and n.startswith(allowed)
            if private_diagnostic and n.startswith(('bart_common.','clip_text.','common_queries','common_attention.','common_norm.','common_tx.','common_alignment.')): active=False
            p.requires_grad_(active); p.grad=None
        self.eval()

    def source_encode(self,bart_hidden=None,source_mask=None,image_wire=None,clip_image=None,clip_text=None):
        if self.modality=='joint':
            if any(v is None for v in (bart_hidden,source_mask,image_wire,clip_image,clip_text)): raise ValueError('Joint requires both sources')
            return super().source_encode(bart_hidden,source_mask,image_wire,clip_image,clip_text)
        if self.modality=='image_only':
            if bart_hidden is not None or source_mask is not None or clip_text is not None: raise ValueError('Text source forbidden in image-only')
            if image_wire is None or clip_image is None or image_wire.shape[1:]!=(224,16): raise ValueError('Image-only source missing')
            b=image_wire.shape[0]; i=unit(image_wire+self.image_tx(image_wire)); t=i.new_zeros(b,32,16)
            memory=torch.cat((self.image_common(image_wire),self.clip_image(clip_image[:,None])),1); pad=None
        else:
            if image_wire is not None or clip_image is not None: raise ValueError('Image source forbidden in text-only')
            if bart_hidden is None or source_mask is None or clip_text is None or bart_hidden.shape[1]!=128: raise ValueError('Text-only source missing')
            b,n,d=bart_hidden.shape; source=bart_hidden*source_mask.to(bart_hidden).unsqueeze(-1)
            h=self.text_fold(source.reshape(b,32,4*d))
            t=unit(self.text_tx(self.text_source(h+self.text_source_position))); i=t.new_zeros(b,224,16)
            memory=torch.cat((self.bart_common(source),self.clip_text(clip_text[:,None])),1)
            pad=torch.cat((~source_mask.bool(),torch.zeros(b,1,device=t.device,dtype=torch.bool)),1)
        q=self.common_queries.expand(b,-1,-1)
        c,_=self.common_attention(q,memory,memory,key_padding_mask=pad,need_weights=False)
        c=self.common_tx(self.common_norm(c+q)); c=torch.cat((unit(c[:,:16]),unit(c[:,16:])),1)
        if self.private_diagnostic: c=torch.zeros_like(c)
        return torch.cat((c,i,t),1)

    def payload(self,full):
        if full.shape[1:]!=(288,16): raise ValueError('Wrong full source layout')
        return full[:,list(indices(self.modality))]

    def expand(self,payload):
        if payload.shape[1:]!=(len(indices(self.modality)),16): raise ValueError('Wrong public compact layout')
        full=payload.new_zeros(payload.shape[0],288,16); full[:,list(indices(self.modality))]=payload
        return full

    def receive_payload(self,payload,reliability,pilot_snr,available):
        full=self.expand(payload)
        if self.modality=='image_only':
            b=full.shape[0]; ok=available.to(full).reshape(b,1,1); full=full*ok
            common=self.common_rx(full[:,:32])
            corrected=full[:,32:256]+self.image_rx(full[:,32:256])
            image_frame=torch.cat((full.new_zeros(b,32,16),corrected,full.new_zeros(b,32,16)),1)*ok
            rel=reliability.to(full).reshape(b,1,1).expand(-1,288,1)
            latent=self.image_reference.receive(image_frame,rel,pilot_snr,available)['latent']
            query=self.image_query(latent.flatten(2).transpose(1,2))
            hint,_=self.image_cross(query,common,common,need_weights=False)
            delta=self.image_residual(hint).transpose(1,2).reshape_as(latent)
            return dict(latent=(latent+delta)*ok.reshape(b,1,1,1),physical_common=full[:,:32],
                common_features=common,common_alignment=self.common_alignment(common.mean(1))*ok[:,0])
        return super().receive(full,reliability,pilot_snr,available,text_only=self.modality=='text_only')

    def bridge_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if not n.startswith(('bart.','image_reference.'))}

    def load_bridge(self,weights):
        named={n:p for n,p in self.named_parameters() if not n.startswith(('bart.','image_reference.'))}
        if set(named)!=set(weights): raise ValueError('Nonexact bridge state')
        with torch.no_grad():
            for n,p in named.items():
                v=weights[n]
                if v.shape!=p.shape or not torch.isfinite(v).all(): raise ValueError('Bad bridge tensor '+n)
                p.copy_(v.to(p))

    def learning_groups(self):
        out={}
        if self.modality!='image_only':
            out.update(text_compression=('text_fold.','text_source.'),text_tx=('text_tx.',),
                text_rx=('text_rx.','text_receiver.','text_expand.'),memory_adapter=('bart_memory_adapter.','bart_memory_norm.'))
        if self.modality!='text_only': out.update(image_tx=('image_tx.',),image_rx=('image_rx.','image_residual.'),image_source=('image_common.','clip_image.'))
        if not self.private_diagnostic: out.update(common_source=('common_attention.','common_tx.'),common_receiver=('common_rx.',))
        return out
