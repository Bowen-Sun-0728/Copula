"""Copula: the OpenAI-compatible proxy over the kernel, with the mock model
(docs/concepts/how-it-works.md, docs/reference/http.md). No GPU:
    PYTHONPATH=src:. python3 -m unittest tests.test_copula_proxy
"""
import asyncio
import json
import os
import shutil
import socket
import tempfile
import unittest
import urllib.error
import urllib.request

from copula.kernel.sdk import Session
from copula.kernel.testing.rig import Rig

from copula.proxy import Proxy
from copula.registry import Registry


def post(url, body, headers=None, timeout=30):
    req = urllib.request.Request(url + '/chat/completions', data=json.dumps(body).encode(),
                                 headers=dict({'Content-Type': 'application/json'},
                                              **(headers or {})), method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            return r.status, hdrs, raw
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, err.read().decode()


def post_raw(url, raw):
    req = urllib.request.Request(url + '/chat/completions', data=raw,
                                 headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as err:
        return err.code


def ask(text, **kw):
    return dict({'model': 'm', 'messages': [{'role': 'user', 'content': text}],
                 'temperature': 0.8, 'max_tokens': 16}, **kw)


class _Base(unittest.IsolatedAsyncioTestCase):
    latency = 0.0

    async def asyncSetUp(self):
        self.rig = await Rig('copula-proxy-%s' % self.latency, latency=self.latency,
                             **self.rig_kw()).start()
        self.dir = tempfile.mkdtemp(prefix='copula-proxy-')
        self.reg = Registry(os.path.join(self.dir, 'registry.db'))
        self.s = await Session.connect(self.rig.socket, trace=os.path.join(self.dir, 't.jsonl'))
        self.proxy = Proxy(self.s, self.reg, {'m': 'actor'}, keepalive=0.05,
                           routes={'proposer': 'per:seed', 'scratch': 'fresh'})
        await self.proxy.start()

    def rig_kw(self):
        return {}

    async def asyncTearDown(self):
        await self.proxy.close()
        await self.s.aclose()
        await self.rig.close()
        self.reg.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    async def call(self, url, body, headers=None):
        loop = asyncio.get_running_loop()
        status, hdrs, raw = await loop.run_in_executor(None, post, url, body, headers)
        data = json.loads(raw) if raw and hdrs.get('content-type', '').startswith(
            'application/json') else raw
        return status, hdrs, data

    def launch(self, exp='e', coords=None, version=None, phase='search'):
        if exp and version is None:
            version = self.reg.match_version(exp, {})[0]
        rec = self.reg.process_run(exp, 'dev', phase, coords, None, version)
        return rec, self.proxy.url(rec)

    @staticmethod
    def text(data):
        return [c['message']['content'] for c in data['choices']]

    def generations(self):
        return sum(self.rig.model.calls.values())


class ProxyTest(_Base):
    async def test_rerun_edit_and_numbering(self):
        r1, u1 = self.launch()
        out1 = []
        for q in ('a', 'b', 'c'):
            st, h, d = await self.call(u1, ask(q))
            self.assertEqual((st, h['x-copula-status']), (200, 'generated'))
            out1 += self.text(d)
        await self.proxy.end([r1['run_id']])
        r2, u2 = self.launch(version='v-edited')            # a new version: slot 1, shared
        out2 = []
        for q in ('a', 'b', 'c'):
            st, h, d = await self.call(u2, ask(q))
            self.assertEqual(h['x-copula-status'], 'hit')
            out2 += self.text(d)
        self.assertEqual(out1, out2)
        r3, u3 = self.launch(version=r1['version'])        # an unchanged rerun: slot 2, fresh
        self.assertEqual(r3['slot'], 2)
        st, h, d = await self.call(u3, ask('a'))
        self.assertEqual(h['x-copula-status'], 'generated')
        self.assertNotEqual(self.text(d), out1[:1])
        self.assertEqual(self.generations(), 4)

    async def test_repeats_in_one_run_never_collapse(self):
        r, u = self.launch(exp=None)
        res = await asyncio.gather(*[self.call(u, ask('same')) for _ in range(4)])
        texts = [self.text(d)[0] for _, _, d in res]
        self.assertEqual(len(set(texts)), 4)
        st, h, d = await self.call(u, ask('same', n=3))
        self.assertEqual(len(set(self.text(d))), 3)
        self.assertEqual(h['x-copula-rank'], '5,6,7')

    async def test_replicates_are_separate(self):
        a, ua = self.launch(coords={'seed': '1'})
        b, ub = self.launch(coords={'seed': '2'}, version=a['version'])
        _, _, da = await self.call(ua, ask('q'))
        _, hb, db = await self.call(ub, ask('q'))
        self.assertEqual(hb['x-copula-status'], 'generated')
        self.assertNotEqual(self.text(da), self.text(db))

    async def test_greedy_is_global_only_with_identity(self):
        _, ua = self.launch()
        _, ub = self.launch(exp=None)
        _, ha, da = await self.call(ua, ask('g', temperature=0))
        _, hb, db = await self.call(ub, ask('g', temperature=0))
        self.assertEqual((ha['x-copula-status'], hb['x-copula-status']), ('generated', 'hit'))
        self.assertEqual(self.text(da), self.text(db))
        self.proxy.identity = False                         # W12: cached per group instead
        _, uc = self.launch(exp=None)
        _, hc, _ = await self.call(uc, ask('g', temperature=0))
        self.assertEqual(hc['x-copula-status'], 'generated')

    async def test_units_by_header(self):
        p1, u1 = self.launch()
        _, h1, d1 = await self.call(u1, ask('x'), {'X-Copula-Coords': 'case=1'})
        self.assertTrue(h1['x-copula-run'].startswith('u-'))
        p2, u2 = self.launch(version='v-other')
        _, h2, d2 = await self.call(u2, ask('x'), {'X-Copula-Coords': 'case=1'})
        _, h3, _ = await self.call(u2, ask('x'), {'X-Copula-Coords': 'case=2'})
        self.assertEqual((h2['x-copula-status'], h3['x-copula-status']), ('hit', 'generated'))
        self.assertEqual(self.text(d1), self.text(d2))
        _, h4, _ = await self.call(u2, ask('x'), {'X-Copula-Coords': 'case=1', 'X-Copula-Unit': '2'})
        self.assertEqual(h4['x-copula-status'], 'generated')   # the 2nd unit of case 1: slot 2

    async def test_roles_route_per_seed(self):
        a, ua = self.launch(coords={'seed': '1'})
        b, ub = self.launch(coords={'seed': '2'}, version=a['version'])
        c, uc = self.launch(coords={'seed': '1'}, version='v-next')
        _, ha, _ = await self.call(ua, ask('reflect'), {'X-Copula-Role': 'proposer'})
        _, hb, _ = await self.call(ub, ask('reflect'), {'X-Copula-Role': 'proposer'})
        _, hc, _ = await self.call(uc, ask('reflect'), {'X-Copula-Role': 'proposer'})
        self.assertEqual([h['x-copula-status'] for h in (ha, hb, hc)],
                         ['generated', 'generated', 'hit'])

    async def test_seeded_requests(self):
        _, u1 = self.launch()
        _, _, d1 = await self.call(u1, ask('s', seed=5, n=2))
        _, u2 = self.launch(version='v-2')
        _, h5, d5 = await self.call(u2, ask('s', seed=5))
        _, h6, d6 = await self.call(u2, ask('s', seed=6))
        _, h7, _ = await self.call(u2, ask('s', seed=7))
        self.assertEqual([h['x-copula-status'] for h in (h5, h6, h7)], ['hit', 'hit', 'generated'])
        self.assertEqual(self.text(d1), self.text(d5) + self.text(d6))    # child i = seed + i

    async def test_errors(self):
        _, u = self.launch()
        st, _, d = await self.call(u, dict(ask('x'), model='other'))
        self.assertEqual((st, d['error']['code']), (404, 'model_not_found'))
        st, _, _ = await self.call(u.replace('/r/', '/r/x'), ask('x'))
        self.assertEqual(st, 401)
        p, up = self.launch()
        await self.call(up, ask('x'), {'X-Copula-Coords': 'case=7'})
        _, uh = self.launch(phase='holdout', version=p['version'])
        st, _, d = await self.call(uh, ask('x'), {'X-Copula-Coords': 'case=7'})
        self.assertEqual((st, d['error']['code']), (409, 'holdout_overlap'))
        st, _, d = await self.call(u, dict(ask('x'), logprobs=True))
        self.assertEqual(st, 400)                          # a field this version refuses


    async def test_ended_and_foreign_runs_are_refused(self):
        r, u = self.launch()
        st, _, _ = await self.call(u, ask('x'))
        self.assertEqual(st, 200)
        await self.proxy.end([r['run_id']])
        st, _, d = await self.call(u, ask('x'))                 # a new attempt would re-read
        self.assertEqual((st, d['error']['code']), (410, 'run_ended'))
        other = self.reg.process_run('e', 'dev', 'search', None, None, 'v-x')
        foreign = '%s/r/%s/v1' % (self.proxy.base, self.reg.token(other['run_id']))
        st, _, _ = await self.call(foreign, ask('x'))           # registered, not served here
        self.assertEqual(st, 401)

    async def test_header_token_wins_over_path(self):
        ra, ua = self.launch()
        rb, ub = self.launch(version='v-b')
        _, h, _ = await self.call(ub, ask('x'), {'X-Copula-Run': self.reg.token(ra['run_id'])})
        self.assertEqual(h['x-copula-run'], ra['run_id'])

    async def test_bad_bodies(self):
        _, u = self.launch()
        loop = asyncio.get_running_loop()
        for raw in (b'[]', b'"x"', b'{bad'):
            self.assertEqual(await loop.run_in_executor(None, post_raw, u, raw), 400)
        st, _, _ = await self.call(u, ask('x', n=0))
        self.assertEqual(st, 400)
        st, h, _ = await self.call(u, ask('x', tools=[{'type': 'web\u2603search'}]))
        self.assertEqual(st, 400)          # a server-side tool: uncached, then refused by the
        self.assertTrue(all(ord(c) < 128 for c in h['x-copula-rule']))   # kernel; clean header

    async def test_fresh_route_is_uncached(self):
        _, ua = self.launch()
        _, ub = self.launch(version='v-b')
        _, ha, da = await self.call(ua, ask('f'), {'X-Copula-Role': 'scratch'})
        _, hb, db = await self.call(ub, ask('f'), {'X-Copula-Role': 'scratch'})
        self.assertNotEqual(self.text(da), self.text(db))
        self.assertEqual((ha['x-copula-status'], hb['x-copula-status']),
                         ('generated', 'generated'))

    async def test_delivered_occurrences_are_forgotten(self):
        _, u = self.launch()
        for i in range(5):
            await self.call(u, ask('m%d' % i), {'Idempotency-Key': 'k%d' % i})
        self.assertEqual((dict(self.proxy._occ), self.proxy._idem), ({}, {}))


    async def test_holdout_inferred_falls_back_declared_fails(self):
        p, u = self.launch()
        _, h0, d0 = await self.call(u, ask('x'), {'X-Copula-Coords': 'case=7'})
        _, uh = self.launch(phase='holdout', version=p['version'])
        st, _, _ = await self.call(uh, ask('x'), {'X-Copula-Coords': 'case=7'})
        self.assertEqual(st, 409)                                   # declared: W5a
        st, h, _ = await self.call(u, ask('x'), {'X-Copula-Coords': 'case=7',
                                                 'X-Copula-Phase': 'holdout',
                                                 'X-Copula-Phase-Source': 'inferred'})
        self.assertEqual(st, 200)                                   # the search unit of case 7:
        self.assertEqual((h['x-copula-run'], h['x-copula-rank']),   # its next occurrence
                         (h0['x-copula-run'], '2'))
        self.assertTrue(any('W5b' in n for n in self.proxy.notes[p['run_id']]))

    async def test_phase_without_coordinates_is_one_holdout_unit(self):
        p, u = self.launch()
        _, h1, d1 = await self.call(u, ask('h'), {'X-Copula-Phase': 'holdout'})
        _, h2, d2 = await self.call(u, ask('h'), {'X-Copula-Phase': 'holdout'})
        _, h3, d3 = await self.call(u, ask('h'))
        self.assertEqual(h1['x-copula-run'], h2['x-copula-run'])
        self.assertNotEqual(h1['x-copula-run'], h3['x-copula-run'])
        self.assertEqual(len({self.text(d)[0] for d in (d1, d2, d3)}), 3)
        p2, u2 = self.launch(version='v-edit')                      # compared versions share
        _, h4, d4 = await self.call(u2, ask('h'), {'X-Copula-Phase': 'holdout'})
        self.assertEqual((h4['x-copula-status'], self.text(d4)), ('hit', self.text(d1)))


class SlowProxyTest(_Base):
    """Retries and buffered streams need a sample that takes a while."""
    latency = 0.4

    async def abandon(self, url, body, headers=None):
        """Send a request and hang up before the reply (a client timeout)."""
        host, port = self.proxy.base[len('http://'):].split(':')
        path = url[len(self.proxy.base):] + '/chat/completions'
        raw = json.dumps(body).encode()
        head = ['POST %s HTTP/1.1' % path, 'Host: x', 'Content-Type: application/json',
                'Content-Length: %d' % len(raw)] + ['%s: %s' % kv for kv in (headers or {}).items()]
        loop = asyncio.get_running_loop()

        def go():
            s = socket.create_connection((host, int(port)))
            s.sendall(('\r\n'.join(head) + '\r\n\r\n').encode() + raw)
            s.close()
        await loop.run_in_executor(None, go)

    async def test_signalled_retry_rereads_the_undelivered_occurrence(self):
        _, u = self.launch()
        await self.abandon(u, ask('r'))
        await asyncio.sleep(0.1)
        st, h, d = await self.call(u, ask('r'), {'x-stainless-retry-count': '1'})
        self.assertEqual(st, 200)
        self.assertIn('retry re-read', h['x-copula-status'])
        self.assertEqual(sum(self.rig.model.calls.values()), 1)
        st, h, _ = await self.call(u, ask('r'), {'x-stainless-retry-count': '1'})
        self.assertEqual(h['x-copula-status'], 'generated')    # nothing undelivered left
        await self.abandon(u, ask('q'))
        await asyncio.sleep(0.1)
        st, h, _ = await self.call(u, ask('q'))                 # no signal: a new occurrence
        self.assertEqual((h['x-copula-status'], h['x-copula-rank']), ('generated', '2'))

    async def test_concurrent_identical_requests_are_not_retries(self):
        _, u = self.launch()
        res = await asyncio.gather(*[self.call(u, ask('c'), {'x-stainless-retry-count': '1'})
                                     for _ in range(3)])
        self.assertEqual(len({self.text(d)[0] for _, _, d in res}), 3)

    async def test_retry_key_includes_the_seed(self):
        _, u = self.launch()
        await self.abandon(u, ask('s', seed=1))
        await asyncio.sleep(0.1)
        _, h, _ = await self.call(u, ask('s', seed=2), {'x-stainless-retry-count': '1'})
        self.assertEqual(h['x-copula-status'], 'generated')

    async def test_retry_after_an_abandoned_retry(self):
        _, u = self.launch()
        await self.abandon(u, ask('t'))
        await asyncio.sleep(0.05)
        await self.abandon(u, ask('t'), {'x-stainless-retry-count': '1'})
        await asyncio.sleep(0.05)
        st, h, _ = await self.call(u, ask('t'), {'x-stainless-retry-count': '2'})
        self.assertIn('retry re-read', h['x-copula-status'])
        self.assertEqual(sum(self.rig.model.calls.values()), 1)

    async def test_buffered_stream_with_keepalive(self):
        _, u = self.launch()
        st, h, raw = await self.call(u, ask('stream me', stream=True,
                                            stream_options={'include_usage': True}))
        self.assertEqual((st, h['content-type']), (200, 'text/event-stream'))
        self.assertLess(raw.index(': keep-alive'), raw.index('data: {'))
        self.assertTrue(raw.rstrip().endswith('data: [DONE]'))
        self.assertIn(': x-copula-status generated', raw)
        events = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith('data: {')]
        self.assertEqual(events[0]['choices'][0]['delta']['role'], 'assistant')
        self.assertIn('usage', events[-1])



class FaultProxyTest(SlowProxyTest):
    """The first sample fails (a stored request error); retries must not re-read the error."""

    def rig_kw(self):
        state = {'n': 0}

        def fault(key, attempt):
            state['n'] += 1
            return 'request' if state['n'] == 1 else None
        return {'fault': fault}

    async def test_signalled_retry_rereads_the_undelivered_occurrence(self):
        pass                                                     # covered by SlowProxyTest

    test_concurrent_identical_requests_are_not_retries = test_signalled_retry_rereads_the_undelivered_occurrence
    test_buffered_stream_with_keepalive = test_signalled_retry_rereads_the_undelivered_occurrence
    test_retry_key_includes_the_seed = test_signalled_retry_rereads_the_undelivered_occurrence
    test_retry_after_an_abandoned_retry = test_signalled_retry_rereads_the_undelivered_occurrence

    async def test_a_delivered_error_is_not_reread(self):
        _, u = self.launch()
        st, _, d = await self.call(u, ask('e'))
        self.assertEqual(st, 400)                                # the stored error sample
        await self.abandon(u, ask('e'))
        await asyncio.sleep(0.1)
        st, h, _ = await self.call(u, ask('e'), {'x-stainless-retry-count': '1'})
        self.assertEqual(st, 200)
        self.assertIn('retry re-read', h['x-copula-status'])


if __name__ == '__main__':
    unittest.main()
