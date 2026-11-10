// Render pinned v10 canvas frames (50 fps) with a local Chromium installation.
const fs=require('fs'),path=require('path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright-core');
(async()=>{const out=path.resolve(process.argv[2]||'v10-demo-frames');fs.mkdirSync(out,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||undefined});
const page=await browser.newPage({viewport:{width:1280,height:720},deviceScaleFactor:1});
await page.goto('file://'+path.join(__dirname,'animation-v10.html'));
await page.evaluate(()=>window.ready);
await page.evaluate(data=>window.facts=data,JSON.parse(fs.readFileSync(path.join(__dirname,'v10-facts.json'),'utf8')));
const fps=await page.evaluate(()=>window.fps),total=Math.round(27*fps);
const preview=process.env.PREVIEW==='1';const samples=preview?[2.8,6.6,11.7,16.0,20.4,24.9]:Array.from({length:total},(_,i)=>i/fps);
for(let i=0;i<samples.length;i++){await page.evaluate(t=>window.render(t),samples[i]);await page.screenshot({path:path.join(out,String(i).padStart(4,'0')+'.png')});}
await browser.close();console.log('Rendered '+samples.length+' frames at '+fps+' fps');})().catch(error=>{console.error(error);process.exit(1)});
