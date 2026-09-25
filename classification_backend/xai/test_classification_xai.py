from .classification_xai import ClassificationXAI

import pandas as pd

top = pd.DataFrame({

    "feature":[

        "APD90",

        "Block_IKr",

        "Peak",

        "IC50_IKr",

        "RMP"

    ],

    "importance":[

        0.91,

        0.72,

        0.41,

        0.35,

        0.22

    ]

})

xai = ClassificationXAI()

result = xai.explain(

    prediction=2,

    confidence=0.93,

    top_features=top,

)

print(result)