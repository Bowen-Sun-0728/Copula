// Injected by `copula run --exp` into Node children through NODE_OPTIONS=--require.
// It records the project files the process reads and writes, in the format of the Python shim
// ('R path' / 'W path' lines appended to $COPULA_READS_FILE.<pid>), the ES modules it imports
// (a module.register load hook, Node >= 18.19), and at exit the files of loaded CommonJS modules
// and the entry script ('M path'). Outside $COPULA_PROJECT, .copula/, node_modules and VCS
// directories nothing is recorded. Never throws into the program.
'use strict';
(function () {
  const root0 = process.env.COPULA_PROJECT;
  const out = process.env.COPULA_READS_FILE;
  if (!root0 || !out) return;
  const fs = require('fs');
  const path = require('path');
  let root;
  try { root = fs.realpathSync(root0); } catch (e) { return; }
  const SKIP = new Set(['.copula', '.git', '.hg', '.svn', '.venv', 'venv', 'env', 'node_modules',
                        '__pycache__', '.cache', '.next', 'dist', 'build']);
  let fd;
  try {
    fd = fs.openSync(out + '.' + process.pid, 'a', 0o600);
  } catch (e) { return; }
  const seen = new Set();
  const write = fs.writeSync;
  const real = fs.realpathSync;

  function inside(p) {
    if (!p.startsWith(root + path.sep)) return false;
    const parts = p.slice(root.length + 1).split(path.sep);
    parts.pop();
    return !parts.some((x) => SKIP.has(x));
  }
  function note(kind, file) {
    try {
      if (typeof file !== 'string') {
        if (file instanceof URL) file = file.pathname;
        else if (Buffer.isBuffer(file)) file = file.toString();
        else return;
      }
      let p = path.resolve(file);
      try { p = real(p); } catch (e) { /* not created yet */ }
      if (!inside(p)) return;
      const key = kind + ' ' + p;
      if (seen.has(key)) return;
      seen.add(key);
      write(fd, key + '\n');
    } catch (e) { /* never break the program */ }
  }
  const WRITE_FLAGS = /^(a|ax|a\+|ax\+|as|as\+|w|wx|w\+|wx\+|r\+|rs\+)$/;
  function writing(flags) {
    if (flags === undefined || flags === null) return false;
    if (typeof flags === 'number') {
      const c = fs.constants;
      return (flags & (c.O_WRONLY | c.O_RDWR | c.O_APPEND | c.O_CREAT | c.O_TRUNC)) !== 0;
    }
    return typeof flags === 'string' && WRITE_FLAGS.test(flags);
  }
  // readFile(p, 'latin1') names an encoding, not flags; only an options object carries flags
  function flagsOf(opts, fallback) {
    if (opts && typeof opts === 'object' && 'flag' in opts) return opts.flag;
    if (opts && typeof opts === 'object' && 'flags' in opts) return opts.flags;
    return fallback;
  }
  function wrap(obj, name, kindOf) {
    const orig = obj[name];
    if (typeof orig !== 'function') return;
    obj[name] = function (file, ...rest) {
      note(kindOf(rest), file);
      return orig.call(this, file, ...rest);
    };
  }
  const readish = (rest) => (writing(flagsOf(rest[0], 'r')) ? 'W' : 'R');
  // open(p[, flags[, mode]][, cb]): a callback in the flags position means read
  const openish = (rest) => (typeof rest[0] !== 'function' && writing(rest[0]) ? 'W' : 'R');
  const writes = () => 'W';
  wrap(fs, 'readFileSync', readish);
  wrap(fs, 'readFile', readish);
  wrap(fs, 'openSync', openish);
  wrap(fs, 'open', openish);
  wrap(fs, 'createReadStream', readish);
  wrap(fs, 'createWriteStream', writes);
  wrap(fs, 'writeFileSync', writes);
  wrap(fs, 'writeFile', writes);
  wrap(fs, 'appendFileSync', writes);
  wrap(fs, 'appendFile', writes);
  if (fs.promises) {
    wrap(fs.promises, 'readFile', readish);
    wrap(fs.promises, 'open', openish);
    wrap(fs.promises, 'writeFile', writes);
    wrap(fs.promises, 'appendFile', writes);
  }
  try {
    const mod = require('module');
    if (typeof mod.register === 'function') {
      const hooks = [
        "import { appendFileSync, realpathSync } from 'node:fs';",
        "import { fileURLToPath } from 'node:url';",
        "import path from 'node:path';",
        'let cfg = null;',
        'export function initialize(data) { cfg = data; }',
        'export async function load(url, context, nextLoad) {',
        "  if (cfg && url.startsWith('file:')) {",
        '    try {',
        '      let p = fileURLToPath(url);',
        '      try { p = realpathSync(p); } catch (e) {}',
        '      if (p.startsWith(cfg.root + path.sep)) {',
        '        const parts = p.slice(cfg.root.length + 1).split(path.sep); parts.pop();',
        "        if (!parts.some((x) => cfg.skip.includes(x))) appendFileSync(cfg.out, 'M ' + p + '\\n');",
        '      }',
        '    } catch (e) {}',
        '  }',
        '  return nextLoad(url, context);',
        '}'].join('\n');
      mod.register('data:text/javascript,' + encodeURIComponent(hooks),
                   require('url').pathToFileURL(__filename).href,
                   {data: {root, out: out + '.' + process.pid, skip: [...SKIP]}});
    }
  } catch (e) { /* older Node: CommonJS only */ }
  process.on('exit', () => {
    try {
      if (process.argv[1]) note('M', process.argv[1]);
      for (const f of Object.keys(require.cache || {})) note('M', f);
    } catch (e) { /* ignore */ }
  });
})();
