from fms.models import get_model
import pytest
import torch
from typing import List

from fms.testing._internal.model_test_suite import (
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
import os

model_dir = os.environ.get("FMS_TESTING_MODEL_DIR", "/tmp/models")
LLAMA_194M = f"{model_dir}/llama-194m"
GRANITE_7B_BASE = f"{model_dir}/granite-7b-base"
GRANITE_8B_CODE_BASE = f"{model_dir}/granite-8b-code-base"
GRANITE_3_8B_CODE_BASE = f"{model_dir}/granite-3-8b-base"

models = [LLAMA_194M, GRANITE_7B_BASE, GRANITE_8B_CODE_BASE, GRANITE_3_8B_CODE_BASE]

class TestAIUModels(
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
):

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    @pytest.fixture(scope="class", autouse=True, params=models)
    def uninitialized_model(self, request):
        model_id = request.param
        # FIXME: better way to do this without environment variable
        os.environ['FMS_MODEL_CONSISTENCY_MODEL_NAME'] = model_id.replace("/", "--")
        aiu_model = get_model(
            "hf_configured",
            model_id,
            device_type="cpu",
            unfuse_weights=True,
            nlayers=3
        )
        torch.compile(aiu_model, backend="sendnn")
        return aiu_model
    
    # ensuring we get the model_id in model_name
    @pytest.fixture(scope="class", autouse=True)
    def signature(self, uninitialized_model) -> List[float]:
        """include this fixture to get a models signature (defaults to what is in tests/resources/expectations)"""
        return self._signature()

    def test_model_unfused(self, model, signature):
        pytest.skip("All AIU models are already unfused")