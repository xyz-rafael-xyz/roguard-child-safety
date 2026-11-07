(function () {
  'use strict';
  const clone = value => JSON.parse(JSON.stringify(value));
  function annotationIssues(row, ontology, complete) {
    const errors = [];
    for (const key of ['uncertain', 'outside_ontology']) {
      if (typeof row[key] !== 'boolean' && !(row[key] === null && !complete)) errors.push('Choose '+key.replaceAll('_', ' ')+'.');
    }
    if (![null, 0, 1].includes(row.review)) errors.push('Invalid review label.');
    if (complete && row.review === null && row.uncertain !== true) errors.push('An unresolved review must be marked uncertain.');
    if (row.world !== null) {
      if (typeof row.world !== 'object' || Array.isArray(row.world) || Object.keys(row.world).length !== Object.keys(ontology).length ||
          Object.entries(ontology).some(([key, values]) => !values.includes(row.world[key]) && !(row.world[key] === '' && !complete))) errors.push('Provide all six valid interpretation fields.');
      if (complete && row.outside_ontology === true) errors.push('An outside-category judgment cannot include an interpretation.');
    } else if (complete && row.uncertain === false && row.outside_ontology === false) errors.push('A resolved in-category judgment needs an interpretation.');
    if (typeof row.notes !== 'string' || row.notes.length > 10000) errors.push('Notes must contain at most 10,000 characters.');
    if (!Array.isArray(row.evidence)) errors.push('Evidence must be a list.');
    else for (const span of row.evidence) {
      if (!span || !['turn','start','end'].every(key => Number.isInteger(span[key])) ||
          span.turn < 0 || span.turn >= row.conversation.turns.length || span.start < 0 || span.end <= span.start ||
          span.end > Array.from(row.conversation.turns[span.turn].text).length) errors.push('Evidence must select text inside a displayed message.');
    }
    return errors;
  }
  function resumePacket(original, returned, ontology) {
    if (!returned || returned.format !== original.format || returned.annotator_id !== original.annotator_id || returned.instructions !== original.instructions ||
        !Array.isArray(returned.items) || returned.items.length !== original.items.length || Object.keys(returned).sort().join() !== Object.keys(original).sort().join()) throw Error('This return does not match the assigned packet.');
    const indexed = new Map(returned.items.map(row => [row.item_id, row]));
    if (indexed.size !== original.items.length) throw Error('Duplicate item in return.');
    const items = original.items.map(source => {
      const row = indexed.get(source.item_id);
      if (!row || Object.keys(row).sort().join() !== Object.keys(source).sort().join() || row.input_digest !== source.input_digest ||
          JSON.stringify(row.conversation) !== JSON.stringify(source.conversation)) throw Error('Conversation or item identity changed.');
      const errors = annotationIssues(row, ontology, false);
      if (errors.length) throw Error(errors[0]);
      return clone(row);
    });
    return {...clone(original), items};
  }
  function spanFromOffsets(text, turn, start, end) {
    if (![turn,start,end].every(Number.isInteger) || turn < 0 || start < 0 || end <= start || end > text.length) throw Error('Select text inside one message.');
    const boundary = n => n === 0 || n === text.length || !(text.charCodeAt(n-1) >= 0xD800 && text.charCodeAt(n-1) <= 0xDBFF && text.charCodeAt(n) >= 0xDC00 && text.charCodeAt(n) <= 0xDFFF);
    if (!boundary(start) || !boundary(end)) throw Error('Selection splits a Unicode character.');
    return {turn, start:Array.from(text.slice(0,start)).length, end:Array.from(text.slice(0,end)).length};
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {annotationIssues, resumePacket, spanFromOffsets};
  if (typeof document === 'undefined') return;
  const byId = id => document.getElementById(id);
  const original = JSON.parse(byId('packet-data').textContent), ontology = JSON.parse(byId('ontology-data').textContent);
  const fieldLabels = {subject:'Who is described?',time:'When does the description apply?',assertion:'How is the claim stated?',source:'How is the account supplied?',relation:'Relation to an earlier claim',concern:'Concern content present, even if negated or fictional?'};
  const optionLabels = {SELF:'The speaker',OTHER:'Another person',FICTIONAL:'A fictional person',CURRENT:'Current',PAST:'Past',HYPOTHETICAL:'Hypothetical',AFFIRMED:'Affirmed',NEGATED:'Negated',UNCERTAIN:'Uncertain claim',RETRACTED:'Retracted claim',DIRECT:'Direct account',REPORTED:'Reported account',FICTION:'Fiction',NEW:'New claim',RETRACTS:'Retracts an earlier claim','0':'No','1':'Yes'};
  let packet = clone(original), index = 0, dirty = false;
  const tell = (message, error=false) => {byId('status').textContent=message;byId('status').className=error?'error':'';};
  const choice = value => value === '' ? null : value === 'true';
  function capture() {
    const row = packet.items[index]; row.review = byId('review').value === '' ? null : Number(byId('review').value);
    row.uncertain = choice(byId('uncertain').value); row.outside_ontology = choice(byId('outside').value);row.notes=byId('notes').value;
    const world = Object.fromEntries(Object.keys(ontology).map(key=>[key,byId('world-'+key).value]));
    row.world = Object.values(world).every(value=>value==='') ? null : world;
  }
  function renderSpans() {
    byId('spans').replaceChildren();
    packet.items[index].evidence.forEach((span, n) => {
      const li=document.createElement('li'), button=document.createElement('button');
      li.append(document.createTextNode('Message '+(span.turn+1)+': “'+Array.from(packet.items[index].conversation.turns[span.turn].text).slice(span.start,span.end).join('')+'”'));
      button.textContent='Remove';button.className='secondary';button.onclick=()=>{packet.items[index].evidence.splice(n,1);dirty=true;renderSpans();};li.append(button);byId('spans').append(li);
    });
  }
  function show() {
    const row=packet.items[index];byId('item').value=String(index);byId('conversation').replaceChildren();
    row.conversation.turns.forEach((turn,n)=>{
      const article=document.createElement('article'), role=document.createElement('div'), text=document.createElement('p');
      article.className=n===row.conversation.turns.length-1?'focal':'';role.className='role';role.textContent=(n+1)+' · '+turn.role;
      text.className='message';text.dataset.turn=String(n);text.textContent=turn.text;article.append(role,text);byId('conversation').append(article);
    });
    byId('review').value=row.review===null?'':String(row.review);byId('uncertain').value=row.uncertain===null?'':String(row.uncertain);byId('outside').value=row.outside_ontology===null?'':String(row.outside_ontology);byId('notes').value=row.notes;
    for (const key of Object.keys(ontology)) byId('world-'+key).value=row.world?.[key]||'';
    byId('previous').disabled=index===0;byId('next').disabled=index===packet.items.length-1;
    const completed=packet.items.filter(r=>annotationIssues(r,ontology,true).length===0).length;
    byId('progress').textContent=completed+' / '+packet.items.length+' complete · '+row.conversation.language.toUpperCase();renderSpans();
  }
  function move(target) {capture();index=target;show();tell('');}
  function download(complete) {
    capture();
    if (complete) {
      const failing=packet.items.findIndex(row=>annotationIssues(row,ontology,true).length);
      if (failing!==-1) {index=failing;show();tell('Item '+(index+1)+': '+annotationIssues(packet.items[index],ontology,true).join(' '),true);return;}
    }
    const blob=new Blob([JSON.stringify(packet,null,2)+'\n'],{type:'application/json'}), url=URL.createObjectURL(blob), anchor=document.createElement('a');
    anchor.href=url;anchor.download='annotations-'+(complete?'completed':'draft')+'.json';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);dirty=false;
    tell(complete?'Completed annotations downloaded. Return this file to the coordinator.':'Draft downloaded. Use Resume to continue later.');
  }
  byId('instructions').textContent=packet.instructions;byId('identity').textContent='Assigned annotator: '+packet.annotator_id;
  packet.items.forEach((row,n)=>{const option=document.createElement('option');option.value=String(n);option.textContent=(n+1)+' / '+packet.items.length;byId('item').append(option);});
  for (const [key,values] of Object.entries(ontology)) {
    const block=document.createElement('div'),label=document.createElement('label'),select=document.createElement('select');label.htmlFor='world-'+key;label.textContent=fieldLabels[key]||key;select.id=label.htmlFor;
    for (const value of ['',...values]) {const option=document.createElement('option');option.value=value;option.textContent=optionLabels[value]||'Unresolved';select.append(option);}
    block.append(label,select);byId('world').append(block);
  }
  for (const id of ['review','uncertain','outside','notes',...Object.keys(ontology).map(k=>'world-'+k)]) byId(id).addEventListener('input',()=>{dirty=true;});
  byId('previous').onclick=()=>move(index-1);byId('next').onclick=()=>move(index+1);byId('item').onchange=()=>move(Number(byId('item').value));
  byId('add-span').onmousedown=event=>event.preventDefault();
  byId('add-span').onclick=()=>{
    try {
      const selected=window.getSelection(),node=selected?.anchorNode,element=node?.parentElement;
      if (!selected || selected.isCollapsed || node!==selected.focusNode || !element?.classList.contains('message') || !byId('conversation').contains(element)) throw Error('Select text within one displayed message first.');
      const span=spanFromOffsets(node.textContent,Number(element.dataset.turn),Math.min(selected.anchorOffset,selected.focusOffset),Math.max(selected.anchorOffset,selected.focusOffset));
      if (!packet.items[index].evidence.some(s=>JSON.stringify(s)===JSON.stringify(span))) packet.items[index].evidence.push(span);
      dirty=true;renderSpans();tell('Evidence added.');
    } catch(error) {tell(error.message,true);}
  };
  byId('save-draft').onclick=()=>download(false);byId('save-final').onclick=()=>download(true);
  byId('resume').onchange=async event=>{
    try {const file=event.target.files[0];if(!file)return;if(file.size>50_000_000)throw Error('Return file is too large.');
      const resumed=resumePacket(original,JSON.parse(await file.text()),ontology);if(dirty&&!window.confirm('Replace unsaved answers with this return?'))return;
      packet=resumed;index=0;dirty=false;show();tell('Saved answers restored.');
    } catch(error) {tell(error.message,true);} finally {event.target.value='';}
  };
  window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});show();
}());
