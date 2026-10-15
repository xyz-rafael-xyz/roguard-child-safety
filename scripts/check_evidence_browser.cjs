/* Local optional browser regression: requires Playwright and a running Studio. */
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const url = new URL(process.env.ROGUARD_TEST_URL || 'http://127.0.0.1:8000/evidence');
  assert.ok(['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname), 'Use a local test server');
  const browser = await chromium.launch({headless:true, ...(process.env.CHROME_BIN ? {executablePath:process.env.CHROME_BIN} : {})});
  try {
    const page = await browser.newPage();
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto(url.href);
    await page.locator('#clear').click();
    const message='  🙂 Îmi este teamă și aș vrea să vorbesc cu cineva.  '.normalize('NFD');
    await page.locator('#turns textarea').fill(message);
    await page.locator('#add').click();
    await page.locator('#turns select').nth(1).selectOption('assistant');
    await page.locator('#turns textarea').nth(1).fill('Te ascult.');
    await page.locator('#run').click();
    await page.waitForFunction(()=>!document.getElementById('export').disabled);
    await page.locator('#cards article button').first().click();
    const selection=await page.locator('#turns textarea').first().evaluate(el=>el.value.slice(el.selectionStart,el.selectionEnd));
    assert.equal(selection,message.trim(),'Reference must preserve original decomposed accents and emoji');
    for(const language of ['uk','ro']) {
      await page.locator('#language').selectOption(language);
      assert.equal(await page.locator('#turns textarea').count(),2);
      assert.equal(await page.locator('#turns textarea').first().inputValue(),message);
      assert.equal(await page.locator('#turns textarea').nth(1).inputValue(),'Te ascult.');
      assert.equal(await page.locator('#turns select').nth(1).inputValue(),'assistant');
      assert.equal(await page.locator('#turns select').first().getAttribute('aria-label'),language==='uk'?'Роль':'Rol');
      assert.equal(await page.locator('#cards article').count(),0);
      assert.equal(await page.locator('#raw').textContent(),'');
      assert.equal(await page.locator('#export').isDisabled(),true);
    }
    await page.locator('#language').selectOption('uk');
    await page.locator('#example').click();
    assert.match(await page.locator('#turns textarea').first().inputValue(),/Пишу/);
    assert.deepEqual(errors,[]);
    console.log('PASS: language switches preserve messages/roles, clear stale reports, localize controls, and retain exact Unicode references');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
