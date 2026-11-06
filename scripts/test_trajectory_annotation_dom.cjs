/* DOM simulation only: this does not replace real-browser visual verification. */
'use strict';
const fs = require('node:fs');
const assert = require('node:assert/strict');
const {JSDOM, VirtualConsole} = require('jsdom');

(async () => {
  const blobs = [], errors = [], console = new VirtualConsole();
  console.on('jsdomError', error => errors.push(error.message));
  const dom = new JSDOM(fs.readFileSync(process.argv[2], 'utf8'), {
    runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.URL.createObjectURL = blob => {blobs.push(blob); return 'blob:test';};
      window.URL.revokeObjectURL = () => {};
      window.HTMLAnchorElement.prototype.click = function () {};
      window.confirm = () => true;
    },
  });
  const w = dom.window, d = w.document, el = id => d.getElementById(id);
  const choose = (id, value) => {el(id).value=value;el(id).dispatchEvent(new w.Event('input',{bubbles:true}));};
  const blobText = blob => new Promise((resolve,reject)=>{const r=new w.FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsText(blob);});
  const resume = async data => {
    Object.defineProperty(el('resume'),'files',{configurable:true,value:[{size:100,text:async()=>JSON.stringify(data)}]});
    el('resume').dispatchEvent(new w.Event('change',{bubbles:true}));
    await new Promise(resolve=>setTimeout(resolve,10));
  };
  const original = JSON.parse(el('packet-data').textContent);
  assert.equal(el('item').options.length, original.items.length);
  assert.equal(el('previous').disabled, true);
  assert.equal(d.querySelectorAll('.message').length, original.items[0].conversation.turns.length);
  choose('review','0');choose('uncertain','true');choose('outside','false');
  choose('notes','Synthetic interface test, not a human annotation.');
  const node=d.querySelector('.message').firstChild, range=d.createRange();range.setStart(node,0);range.setEnd(node,3);
  w.getSelection().removeAllRanges();w.getSelection().addRange(range);el('add-span').click();
  assert.equal(el('spans').children.length,1);
  el('save-draft').click();assert.equal(blobs.length,1);
  const draft=JSON.parse(await blobText(blobs[0]));assert.equal(draft.items[0].evidence.length,1);
  el('next').click();assert.equal(el('item').value,'1');
  await resume(draft);assert.equal(el('item').value,'0');assert.equal(el('notes').value,draft.items[0].notes);
  const tampered=structuredClone(draft);tampered.items[0].conversation.turns[0].text+='changed';
  await resume(tampered);assert.match(el('status').textContent,/changed/i);assert.equal(el('notes').value,draft.items[0].notes);
  el('save-final').click();assert.equal(blobs.length,1);assert.match(el('status').textContent,/Item 2/);
  for(let i=0;i<original.items.length;i++) {
    el('item').value=String(i);el('item').dispatchEvent(new w.Event('change',{bubbles:true}));
    choose('review','0');choose('uncertain','true');choose('outside','false');
  }
  el('save-final').click();assert.equal(blobs.length,2);
  const completed=JSON.parse(await blobText(blobs[1]));
  assert.equal(completed.items.length,original.items.length);assert.match(el('status').textContent,/Completed annotations downloaded/);
  fs.writeFileSync(process.argv[3],JSON.stringify(completed));
  assert.deepEqual(errors,[]);w.close();
  process.stdout.write(JSON.stringify({complete:true,items:completed.items.length,checks:['render','evidence selection','draft download','resume','input-tamper rejection','incomplete-final rejection','completed download'],real_browser_verified:false})+'\n');
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
