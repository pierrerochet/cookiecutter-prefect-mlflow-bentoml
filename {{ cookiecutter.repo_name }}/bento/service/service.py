from typing import Any, Dict, List

import bentoml
from bentoml.io import JSON, NumpyNdarray
from numpy.typing import NDArray

runner = bentoml.sklearn.get("pmb-model:latest").to_runner()

svc = bentoml.Service("pmb-model", runners=[runner])


@svc.api(input=NumpyNdarray(), output=JSON())
def predict(input: NDArray[Any]) -> Dict[str, Any]:
    return {"predictons": svc.predict.run(input)}
