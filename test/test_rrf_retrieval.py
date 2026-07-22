import unittest

from pipeline.retrieve_answer import reciprocal_rank_fusion


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_reciprocal_rank_fusion_rewards_results_returned_by_multiple_queries(self):
        result_groups = [
            [{"uuid": "a", "content": "first"}, {"uuid": "b", "content": "second"}],
            [{"uuid": "b", "content": "second"}, {"uuid": "a", "content": "first"}],
            [{"uuid": "b", "content": "second"}],
        ]

        results = reciprocal_rank_fusion(result_groups)

        self.assertEqual([result["uuid"] for result in results], ["b", "a"])
        self.assertGreater(results[0]["rrf_score"], results[1]["rrf_score"])


if __name__ == "__main__":
    unittest.main()
