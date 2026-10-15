import unittest

from selected_identity_contract import contract


class IdentityContractChecks(unittest.TestCase):
    def test_missing_previous_id_rejects_even_with_high_model_score(self):
        self.assertEqual(contract(17,[20,21],"VALID",.999,.5),{"status":"TARGET_LOST","selected_id":None})

    def test_wrong_initial_identity_can_still_be_held(self):
        # No semantic ground truth argument exists: a live wrong binding persists.
        self.assertEqual(contract(20,[20,21],"VALID",.999,.9),{"status":"SELECTED_HELD_ID","selected_id":20})

    def test_confidence_and_schema_rejection_do_not_switch_identity(self):
        self.assertEqual(contract(20,[20,21],"VALID",.6,.7),{"status":"AMBIGUOUS","selected_id":None})
        self.assertEqual(contract(20,[20,21],"UNSUPPORTED_QUERY",.999,.5),{"status":"UNSUPPORTED_QUERY","selected_id":None})


if __name__=="__main__":unittest.main()
