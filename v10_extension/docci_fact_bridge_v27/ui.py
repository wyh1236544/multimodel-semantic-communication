"""Self-contained PRIVATE offline forms. No fetch, external assets, model or auto-labeling."""
import html
import json
from core import CATEGORIES, VISIBILITY, LABELS

STYLE = '''body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;max-width:1150px;margin:28px auto;padding:0 20px;color:#18232b}h1{font-size:25px}section{border:1px solid #ccd5dc;padding:16px;margin:16px 0;background:#fafcfd}textarea{width:100%;min-height:65px;font:15px/1.5 monospace}input[type=text]{min-width:230px;padding:6px}select,button{padding:7px;margin:5px}button{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}.warn{background:#fff6e8;padding:12px;border-left:4px solid #bc7b20}.fact{border-top:1px solid #ccd5dc;padding:12px 0}.bar{position:sticky;top:0;background:white;border-bottom:1px solid #aaa;padding:8px;z-index:1}.muted{color:#52626c;font-size:14px}label{margin-right:10px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}a{color:#235d85}'''


def embedded(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def page(title, body, payload=None, script=''):
    out = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    out += '<title>' + html.escape(title) + '</title><style>' + STYLE + '</style><body><h1>' + html.escape(title) + '</h1>' + body
    if payload is not None:
        out += '<script type="application/json" id="payload">' + embedded(payload) + '</script><script>' + COMMON + script + '</script>'
    return out + '</body></html>'


COMMON = r'''
'use strict';
const P = JSON.parse(document.getElementById('payload').textContent);
let state = JSON.parse(JSON.stringify(P.template));
const root = document.getElementById('cases');
function el(tag,text,parent){const n=document.createElement(tag);if(text!==null)n.textContent=text;if(parent)parent.appendChild(n);return n;}
function box(value,parent,change){const n=el('textarea',null,parent);n.value=value;n.addEventListener('input',()=>change(n.value));return n;}
function choose(values,value,parent,change){const n=el('select',null,parent);for(const [key,label] of values){const o=el('option',label,n);o.value=key;}n.value=value;n.addEventListener('change',()=>change(n.value));return n;}
function head(){document.getElementById('reviewer').value=state.reviewer;document.getElementById('human').checked=state.human_attested;const n=document.getElementById('exposure');if(n)n.value=state.prior_candidate_exposure;}
document.getElementById('reviewer').addEventListener('input',e=>{state.reviewer=e.target.value;});
document.getElementById('human').addEventListener('change',e=>{state.human_attested=e.target.checked;});
document.getElementById('export').addEventListener('click',()=>{const b=new Blob([JSON.stringify(state,null,2)],{type:'application/json'});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download=P.filename;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});
document.getElementById('import').addEventListener('change',e=>{const f=e.target.files[0];if(!f)return;if(f.size>16000000){alert('文件过大');return;}const r=new FileReader();r.onload=()=>{try{const v=JSON.parse(r.result);const binding=state.kind==='references'?'prepared_sha256':'review_sha256';if(v.schema!==state.schema||v.kind!==state.kind||v[binding]!==state[binding]||!Array.isArray(v.items)||v.items.length!==state.items.length)throw Error('文件不属于本次核验');for(let i=0;i<v.items.length;i++){const id=state.kind==='references'?'reference_id':'case_id';if(v.items[i][id]!==state.items[i][id])throw Error('样本顺序或ID不一致');}state=v;head();render();}catch(err){alert(String(err));}};r.readAsText(f);});
head();
'''

CONTROLS = '''<div class="bar"><label>审核者代号 <input type="text" id="reviewer"></label><label><input type="checkbox" id="human">我已人工核对本次提交，未把模型分数当事实判定</label><button id="export">下载当前填写的 JSON</button><label>继续填写：<input type="file" id="import" accept=".json"></label></div><p class="muted">不会联网或自动保存。可随时下载进度，下次导入继续。未完成文件可以保存，但正式验收会拒绝空项。网页内容仅供内部研究，不要上传公开仓库。</p><div id="cases"></div>'''


def reference_form(material, template):
    payload = dict(template=template, references=material['references'], filename='V27_references_human.json',
                   categories=list(CATEGORIES), visibility=list(VISIBILITY))
    script = r'''
const exposure=el('div',null,document.querySelector('.bar'));el('span','此前是否看过这些样本的候选结果：',exposure);const exposureSelect=choose([['unknown','不确定'],['yes','看过'],['no','未看过']],state.prior_candidate_exposure,exposure,x=>{state.prior_candidate_exposure=x;});exposureSelect.id='exposure';
function render(){root.replaceChildren();state.items.forEach((r,i)=>{const sec=el('section',null,root);el('h2',`${i+1}/32 · ${r.reference_id}`,sec);el('pre',P.references[i].source_text,sec);el('p','只根据原文独立列出3–30条原子事实；此阶段不看任何候选输出。potentially_visual仅表示可能可见，尚未对照图片验证。',sec);r.facts.forEach((f,j)=>{const d=el('div',null,sec);d.className='fact';el('strong',f.fact_id,d);choose(P.categories.map(x=>[x,x]),f.category,d,x=>{f.category=x;});choose(P.visibility.map(x=>[x,x]),f.visibility,d,x=>{f.visibility=x;});el('p','事实：例如“车是红色的”，不要把整段描述写成一条事实。',d);box(f.statement,d,x=>{f.statement=x;});el('p','原文中的逐字证据（复制粘贴连续片段）：',d);box(f.source_quote,d,x=>{f.source_quote=x;});const del=el('button','删除此事实',d);del.onclick=()=>{r.facts.splice(j,1);render();};});const add=el('button','添加一条事实',sec);add.onclick=()=>{let j=1;while(r.facts.some(x=>x.fact_id==='f'+String(j).padStart(2,'0')))j++;r.facts.push({fact_id:'f'+String(j).padStart(2,'0'),category:'objects',visibility:'potentially_visual',statement:'',source_quote:''});render();};});}
render();
'''
    return page('V27 第一步：仅看原文，先冻结事实清单',
        '<p class="warn">没有自动事实标签。不要先看方法输出再挑事实；原子性和原文是否支持须人工确认。至少3条事实只是最低覆盖要求，不代表足够完整。</p>' + CONTROLS, payload, script)


def rating_form(material, references, template):
    # Do not embed method IDs, WER or the private method mapping into the rater page.
    refs = {r['reference_id']: r['facts'] for r in references['items']}
    public = [dict(case_id=r['case_id'], reference_id=r['reference_id'], text=r['text'],
                   facts=refs[r['reference_id']]) for r in material['candidates']]
    payload = dict(template=template, cases=public, filename='V27_ratings_human.json')
    script = r'''
function render(){root.replaceChildren();const picker=el('select',null,root);P.cases.forEach((r,i)=>{const o=el('option',`${i+1}/160 · ${r.case_id} · ${r.reference_id}`,picker);o.value=String(i);});const display=el('div',null,root);function show(i){display.replaceChildren();const c=P.cases[i],r=state.items[i];const sec=el('section',null,display);el('h2',`${c.case_id} · ${c.reference_id}`,sec);el('pre',c.text,sec);el('p','对每条冻结事实判定：retained=保留；contradicted=明确矛盾；missing=未提及；uncertain=无法判断。保留/矛盾必须复制候选中的证据。',sec);r.judgments.forEach((j,k)=>{const f=c.facts[k],d=el('div',null,sec);d.className='fact';el('strong',`${f.fact_id} · ${f.category} · ${f.statement}`,d);el('p','原文证据：'+f.source_quote,d);choose([['','尚未判定'],['retained','保留'],['contradicted','明确矛盾'],['missing','未提及'],['uncertain','无法判断']],j.label,d,x=>{j.label=x;});el('p','候选逐字证据（missing必须留空）：',d);box(j.candidate_quote,d,x=>{j.candidate_quote=x;});el('p','备注（uncertain必填）：',d);box(j.note,d,x=>{j.note=x;});});el('h3','原文未支持的新增事实',sec);el('p','不等于图片中必然为假；请记录候选中新增的、原文无法支持的原子事实。确认没有时保持列表为空。',sec);r.unsupported.forEach((u,j)=>{const d=el('div',null,sec);d.className='fact';el('p','新增事实：',d);box(u.statement,d,x=>{u.statement=x;});el('p','候选逐字证据：',d);box(u.candidate_quote,d,x=>{u.candidate_quote=x;});el('p','原文为何未支持：',d);box(u.note,d,x=>{u.note=x;});const b=el('button','删除此条新增事实',d);b.onclick=()=>{r.unsupported.splice(j,1);show(i);};});const add=el('button','添加一条未获支持的新增事实',sec);add.onclick=()=>{r.unsupported.push({statement:'',candidate_quote:'',note:''});show(i);};const prev=el('button','上一条',sec);prev.onclick=()=>{if(i>0){picker.value=String(i-1);show(i-1);}};const next=el('button','下一条',sec);next.onclick=()=>{if(i+1<P.cases.length){picker.value=String(i+1);show(i+1);}};}picker.onchange=()=>show(Number(picker.value));show(0);}
render();
'''
    return page('V27 第二步：隐藏方法名称的候选事实审核',
        '<p class="warn">已冻结参考事实。仅按候选是否表达该事实判定，不按措辞是否相同判定。文本风格可能暴露方法，随机编号不是严格盲法保证。unknown不会自动计成成功。</p>' + CONTROLS, payload, script)


def index(stage, result=None):
    if stage == 'prepared':
        return page('V27：已准备，等待人工事实清单',
            '<p>32条已有DEV，图文AWGN 0 dB。无模型加载、无新推理、无训练。</p><p><a href="references.html">打开原文事实填写页</a></p><p>下载填写后的JSON，使用seal-facts创建新的审核目录。只有下一步网页才展示候选输出。</p><p class="warn">分析者的literal_overlap_hints.json仅为词语重合线索，不是事实或指标；冻结参考前不要给事实选择者看。</p>')
    if stage == 'review':
        return page('V27：事实已冻结，等待候选人工判定',
            '<p><a href="rate_candidates.html">打开160份候选审核页</a></p><p>下载填写后的JSON，再运行score。程序不自动赋标签，不自动开训。方法对照表未嵌入审核网页，但由文本风格仍可能猜出方法。</p>')
    e = html.escape
    rs = result['groups']
    body = '<p class="warn">这是历史输出的人工源文本事实审核，不是新模型成绩。不能把原文未支持视为已证实的图像幻觉，也没有自动训练批准或语义成功。</p>'
    body += '<table><tr><th>方法</th><th>事实保留下界</th><th>上界（含待定）</th><th>矛盾比例</th><th>每描述未支持新增数</th></tr>'
    for r in rs:
        body += '<tr>' + ''.join('<td>' + e(str(v)) + '</td>' for v in
            (r['method'], '{:.2%}'.format(r['macro_retention_lower']), '{:.2%}'.format(r['macro_retention_upper']),
             '{:.2%}'.format(r['macro_contradiction_fraction']), '{:.3f}'.format(r['mean_unsupported_added_facts']))) + '</tr>'
    body += '</table><h2>是否有必要提出一轮小纠错实验？</h2><pre>' + e(json.dumps(result['opportunity'], ensure_ascii=False, indent=2)) + '</pre>'
    body += '<p>8条描述/16条可补事实是事先固定的探索性机会筛查阈值，不是文献阈值、统计显著性或训练授权。</p><details><summary>完整配对区间与限制</summary><pre>' + e(json.dumps(result, ensure_ascii=False, indent=2)) + '</pre></details>'
    return page('V27：人工审核结果与下一步机会筛查', body)
