"""A named care role never grants access without an explicit matching scope."""

import unittest

from roguard.cli import assess_json


class RoleTermTests(unittest.TestCase):
    def test_romanian_and_ukrainian_role_names_do_not_grant_access(self):
        for language, role in (("ro", "tutore"), ("uk", "піклувальник")):
            with self.subTest(language=language):
                report = assess_json({"language": language, "routing": {
                    "principal": "minor", "item": "element-fictiv", "purpose": "A",
                    "proposed_recipient": role, "recipient_roles": ["minor", role],
                    "allowed_by_scope": [{"principal": "minor", "item": "element-fictiv",
                                          "purpose": "A", "recipients": ["minor"]}],
                }})
                self.assertTrue(report["review_suggested"])
                self.assertEqual(report["findings"][0]["reason_codes"],
                                 ["PRINCIPAL_SCOPE_CONFLICT"])
                self.assertFalse(report["caller_facts_verified"])


if __name__ == "__main__":
    unittest.main()
