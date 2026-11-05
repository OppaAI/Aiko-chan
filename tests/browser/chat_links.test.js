'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('interface/webui/static/script.js', 'utf8');
const encode = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const decode = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&amp;/g, '&');
const ctx = vm.createContext({URL, location: {href: 'https://aiko.test/'}, document: {
  createElement() {
    return {set textContent(s) {this.html = encode(s);}, get innerHTML() {return this.html;},
      set innerHTML(s) {this.html = s;}, get value() {return decode(this.html);}};
  }
}});
vm.runInContext(source.slice(source.indexOf('function esc('), source.indexOf('function isControlTokenChunk(')), ctx);
assert.equal(ctx.parseMarkdown('[click](javascript:alert)'), 'click');
assert.equal(ctx.parseMarkdown('[click](data:text/html,evil)'), 'click');
assert.equal(ctx.parseMarkdown('[click](java\nscript:evil)'), 'click');
assert.ok(ctx.parseMarkdown('[site](https://example.com/)').includes('href="https://example.com/"'));
assert.ok(!ctx.parseMarkdown('[site](https://example.com/" onclick="evil)').includes(' onclick="evil'));
assert.equal(ctx.parseMarkdown('<script>evil</script>'), '&lt;script&gt;evil&lt;/script&gt;');
console.log('chat link checks passed');
vm.runInContext(source.slice(source.indexOf('function isControlTokenChunk('), source.indexOf('function renderAikoContent(')), ctx);
{
  const text = 'I missed you (more than I expected). **Stay a little.** [Read this](https://example.com/).';
  const result = ctx.parseAikoMessage(text);
  assert.equal(result.dialogueText, text);
  assert.equal(result.nonVerbalText, '');
  const action = ctx.parseAikoMessage('*smiles* Welcome home (I am glad).');
  assert.equal(action.nonVerbalText, '*smiles*');
  assert.equal(action.dialogueText, 'Welcome home (I am glad).');
  const legacy = ctx.parseAikoMessage('EMOTION: happy Welcome home.');
  assert.equal(legacy.dialogueText, 'Welcome home.');
}
console.log('chat prose checks passed');
