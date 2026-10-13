"""Loopback Decision Studio behavior and request boundary."""

import json
import re
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from roguard.studio import make_server


class StudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"
        with urlopen(cls.origin + "/", timeout=5) as response:
            page = response.read().decode("utf-8")
            cls.headers = response.headers
        cls.token = re.search(r'data-token="([a-f0-9]{64})"', page).group(1)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, endpoint, payload, *, token=None, origin=None, host=None):
        headers = {"Content-Type": "application/json", "Origin": origin or self.origin,
                   "X-RoGuard-Token": token if token is not None else self.token}
        if host is not None:
            headers["Host"] = host
        data = json.dumps(payload).encode("utf-8")
        try:
            with urlopen(Request(self.origin + endpoint, data=data, headers=headers,
                                 method="POST"), timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def test_conversation_workspace_and_api_boundaries(self):
        with urlopen(self.origin + "/conversation", timeout=5) as response:
            page=response.read().decode()
            self.assertIn('data-token="'+self.token+'"',page)
            self.assertNotIn('__ROGUARD_TOKEN__',page)
        from test_conversation import sample
        payload=sample()
        status,report=self.request('/api/conversation',payload)
        self.assertEqual(status,200)
        self.assertFalse(report['text_in_report'])
        self.assertFalse(report['external_action_taken'])
        self.assertEqual(self.request('/api/conversation',payload,token='wrong')[0],403)
        self.assertEqual(self.request('/api/conversations',[payload])[0],200)
        self.assertEqual(self.request('/api/compare-reports',{'before':[report],'after':[report]})[1]['decision_changes'],[])
        self.assertEqual(self.request('/api/conversation',{'language':'ro','turns':[]})[0],400)
        payload['turns'][0]['role']=[]
        self.assertEqual(self.request('/api/conversation',payload)[0],400)

    def test_interaction_lab_offline_routes_and_invalid_authority(self):
        for asset in ('/lab', '/lab.js', '/lab.css'):
            with urlopen(self.origin + asset, timeout=5) as response:
                data=response.read().decode()
                self.assertTrue(data)
                self.assertNotIn('__ROGUARD_TOKEN__',data)
                if asset=='/lab':self.assertIn(self.token,data)
        status,report=self.request('/api/lab',{'language':'uk','target':'stale','reduce':True})
        self.assertEqual(status,200)
        self.assertEqual(report['run']['requests'][-1]['outcome'],'BLOCK')
        self.assertEqual(report['reproducer']['status'],'REDUCED')
        self.assertEqual(self.request('/api/lab',{},token='wrong')[0],403)
        self.assertEqual(self.request('/api/lab',{'scenario':[]})[0],400)
        self.assertEqual(self.request('/api/lab',{'callback':'run-code'})[0],400)
        self.assertEqual(self.request('/api/lab/compare',{'language':'ro','seed':17})[0],200)
        self.assertEqual(self.request('/api/lab/consistency',[])[0],400)

    def test_security_headers_and_packaged_assets(self):
        self.assertEqual(self.headers["Cache-Control"], "no-store")
        self.assertIn("default-src 'none'", self.headers["Content-Security-Policy"])
        for asset in ("/studio.js", "/studio.css"):
            with urlopen(self.origin + asset, timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())

    def test_check_explore_and_contrast_share_local_contract_rules(self):
        before = {"language": "ro", "gate": {"review_owner": "reviewer",
                  "model_output_parsed": True, "accept_unparsed": False,
                  "external_action_branch": False, "human_approval_required": True}}
        status, check = self.request("/api/assess", before)
        self.assertEqual(status, 200)
        self.assertEqual(check["findings"][0]["status"], "pass")
        status, exploration = self.request("/api/explore", before)
        self.assertEqual(status, 200)
        self.assertTrue(any(row["path"] == "/gate/accept_unparsed"
                            for row in exploration["decision_changes"]))
        after = json.loads(json.dumps(before))
        after["gate"]["accept_unparsed"] = True
        status, contrast = self.request("/api/contrast", {"before": before, "after": after})
        self.assertEqual(status, 200)
        self.assertEqual(contrast["changed_path"], "/gate/accept_unparsed")

    def test_guided_v1_endpoint_uses_corrected_rules_in_both_languages(self):
        for language in ("ro", "uk"):
            with self.subTest(language=language):
                payload = {"language": language,
                           "disclosure": {"source_role": "minor", "safety_or_support_anchor": False,
                                          "indirect_or_repeated_support_pattern": False,
                                          "explicit_support_request": False,
                                          "direct_safety_statement": True,
                                          "retracted_or_denied": True},
                           "routing": {"principal": "minor", "item": "symbol", "purpose": "A",
                                       "proposed_recipient": "guardian", "recipient_roles": ["minor", "guardian"],
                                       "allowed_by_scope": [{"principal": "minor", "item": "symbol",
                                                             "purpose": "A", "recipients": ["guardian"]}],
                                       "recipient_may_be_source_of_concern": "unknown"},
                           "support": {"applicable_fields": ["next_step", "no_pressing_for_details"],
                                       "passed_fields": ["next_step"]}}
                status, report = self.request("/api/v1/assess", payload)
                self.assertEqual(status, 200)
                self.assertEqual(report["language"], language)
                self.assertEqual([check["decision_code"] for check in report["checks"]],
                                 ["REVIEW_SUPPORT_SIGNAL", "REVIEW_ROUTE", "REVIEW_RESPONSE"])
                self.assertFalse(report["external_action_taken"])
                self.assertTrue(report["caller_facts_unverified"])
                status, sensitivity = self.request("/api/v1/explore", payload)
                self.assertEqual(status, 200)
                self.assertTrue(sensitivity["decision_changes"])
                self.assertFalse(sensitivity["external_action_taken"])
                status, envelope = self.request("/api/v1/envelope", payload)
                self.assertEqual(status, 200)
                self.assertEqual(envelope["language"], language)
                self.assertEqual([row["category"] for row in envelope["categories"]],
                                 ["D1", "R1", "S1"])
                self.assertEqual(envelope["categories"][1]["decision_stability"], "sensitive")

    def test_guided_v1_endpoint_rejects_invalid_and_cross_origin_input(self):
        self.assertEqual(self.request("/api/v1/assess", {"language": "ro"})[0], 400)
        self.assertEqual(self.request("/api/v1/assess", {"language": "uk", "support": {}},
                                      token="wrong")[0], 403)

    def test_v2_audit_endpoint_handles_partial_support_facts(self):
        for language in ("ro", "uk"):
            status, report = self.request("/api/v2/audit", {"language": language,
                "support": {"field_states": {"next_step": "passed",
                                              "no_pressing_for_details": "unknown"}}})
            self.assertEqual(status, 200)
            self.assertEqual(report["schema_version"], "2.0")
            self.assertEqual(report["language"], language)
            self.assertIn("/support/field_states/no_pressing_for_details",
                          report["categories"][0]["unknown_paths"])
            self.assertFalse(report["external_action_taken"])
        self.assertEqual(self.request("/api/v2/audit", {"language": "ro",
                         "support": {"field_states": {"next_step": "invented"}}})[0], 400)

    def test_cross_origin_token_and_host_are_rejected(self):
        payload = {"language": "uk", "gate": {"review_owner": "reviewer",
                   "model_output_parsed": True, "accept_unparsed": False,
                   "external_action_branch": False, "human_approval_required": True}}
        self.assertEqual(self.request("/api/assess", payload, token="wrong")[0], 403)
        self.assertEqual(self.request("/api/assess", payload, origin="http://evil.example")[0], 403)
        self.assertEqual(self.request("/api/assess", payload, host="evil.example")[0], 403)

    def test_malformed_input_error_does_not_reflect_values(self):
        payload = {"language": "ro", "gate": {"private child text": "invalid"}}
        status, result = self.request("/api/assess", payload)
        self.assertEqual(status, 400)
        self.assertNotIn("private child text", json.dumps(result))

    def test_raw_decimal_input_keeps_exact_word_cap(self):
        raw = ('{"language":"ro","readability":{"declared_age":12,'
               '"measured_words":9007199254740993.0,"max_words":9007199254740992}}')
        request = Request(self.origin + "/api/assess", data=raw.encode("utf-8"), method="POST",
                          headers={"Content-Type": "application/json", "Origin": self.origin,
                                   "X-RoGuard-Token": self.token})
        with urlopen(request, timeout=5) as response:
            report = json.load(response)
        self.assertEqual(report["findings"][0]["reason_codes"], ["WORD_CAP_EXCEEDED"])

    def test_deep_json_is_rejected_and_server_remains_available(self):
        raw = "[" * 12000 + "0" + "]" * 12000
        request = Request(self.origin + "/api/assess", data=raw.encode("utf-8"), method="POST",
                          headers={"Content-Type": "application/json", "Origin": self.origin,
                                   "X-RoGuard-Token": self.token})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 400)
        with urlopen(self.origin + "/", timeout=5) as response:
            self.assertEqual(response.status, 200)


if __name__ == "__main__":
    unittest.main()
