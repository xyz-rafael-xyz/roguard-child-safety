// Usage: node render.cjs OUTPUT_DIR (requires playwright-core and Chrome).
const fs=require('fs'),path=require('path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright-core');
(async()=>{const out=path.resolve(process.argv[2]||'demo-frames');fs.mkdirSync(out,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||undefined});
const page=await browser.newPage({viewport:{width:1280,height:720},deviceScaleFactor:1});
await page.goto('file://'+path.join(__dirname,'animation-v9.html'));
await page.evaluate(data=>window.facts=data,JSON.parse(fs.readFileSync(path.join(__dirname,'facts.json'),'utf8')));
const samples=process.env.PREVIEW==='1'?[2,8,14,20,25]:Array.from({length:560},(_,i)=>i/20);
for(let i=0;i<samples.length;i++){await page.evaluate(t=>window.render(t),samples[i]);await page.screenshot({path:path.join(out,String(i).padStart(4,'0')+'.png')});}
await browser.close();console.log('Rendered '+samples.length+' frames');})().catch(error=>{console.error(error);process.exit(1)});
