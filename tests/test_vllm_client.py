from vllm_client import VLLMClient


def test_cache_key_distinguishes_sampling_seeds():
    client = VLLMClient()
    first = client._get_cache_key("system", "word", "model", 0.8, 1)
    second = client._get_cache_key("system", "word", "model", 0.8, 2)
    repeated = client._get_cache_key("system", "word", "model", 0.8, 1)
    assert first != second
    assert first == repeated
