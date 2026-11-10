// Render pinned v10 canvas frames with a local Chromium installation.
const fs=require('fs'),path=require('path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright-core');
(async()=>{const out=path.resolve(process.argv[2]||'v10-demo-frames');fs.mkdirSync(out,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||undefined});
const page=await browser.newPage({viewport:{width:1280,height:720},deviceScaleFactor:1});
await page.goto('file://'+path.join(__dirname,'animation-v10.html'));
await page.evaluate(data=>window.facts=data,JSON.parse(fs.readFileSync(path.join(__dirname,'v10-facts.json'),'utf8')));
const preview=process.env.PREVIEW==='1';const samples=preview?[1.6,6.6,11.7,16.0,20.4,24.9]:Array.from({length:540},(_,i)=>i/20);
for(let i=0;i<samples.length;i++){await page.evaluate(t=>window.render(t),samples[i]);await page.screenshot({path:path.join(out,String(i).padStart(4,'0')+'.png')});}
await browser.close();console.log('Rendered '+samples.length+' frames');})().catch(error=>{console.error(error);process.exit(1)});
