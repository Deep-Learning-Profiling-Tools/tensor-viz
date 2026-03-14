import numpy as np
import tensor_viz
from tensor_viz.bundle import create_session_bundle

'''
'''

x = (
    np.arange(2048)
      .reshape(2,2,2,2,2,2,2,2,2,2,2)
      .transpose(7, 0, 1, 9, 2, 3, 4, 8, 5, 6, 10)
      .reshape(128, 16)
)
#import matplotlib.pyplot as plt
#plt.imshow(x)
#plt.show()
#x = np.arange(2048).reshape(2,2,2,2, 2,2,2,2,2, 2,2)#.transpose(-1,-8,-4,-11,-2,-3,-5,-6,-7,-9,-10)
tensor_viz.viz(x)

#SHAPE = (3, 4)
#BASE = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
#BLUE_RGBA = [0, 90, 255, 255]
#BLUE_HS = [240, 1]
#HIGHLIGHTS = {(0, 1), (1, 2), (2, 0)}
#
#
#def dense_rgba() -> list[float]:
#    values: list[float] = []
#    for row in range(SHAPE[0]):
#        for col in range(SHAPE[1]):
#            values.extend(BLUE_RGBA if (row, col) in HIGHLIGHTS else [144, 164, 174, 255])
#    return values
#
#
#def dense_hs() -> list[float]:
#    values: list[float] = []
#    for row in range(SHAPE[0]):
#        for col in range(SHAPE[1]):
#            values.extend(BLUE_HS if (row, col) in HIGHLIGHTS else [0, 0])
#    return values
#
#
#tensors = {
#    "Dense RGBA": BASE,
#    "Coords RGBA": BASE + 100,
#    "Region RGBA": BASE + 200,
#    "Dense HS": BASE + 300,
#    "Coords HS": BASE + 400,
#    "Region HS": BASE + 500,
#}
#
#bundle = create_session_bundle(
#    tensors,
#    color_instructions={
#        # TODO: tensor view + separate colorbar each tensor here
#        "tensor-1": [{"mode": "rgba", "kind": "dense", "values": dense_rgba()}],
#        "tensor-2": [{"mode": "rgba", "kind": "coords", "coords": [list(coord) for coord in sorted(HIGHLIGHTS)], "color": BLUE_RGBA}],
#        "tensor-3": [{"mode": "rgba", "kind": "region", "base": [0, 0], "shape": [3, 2], "jumps": [1, 2], "color": BLUE_RGBA}], # TODO: jumps rendered as [2, 1]
#        "tensor-4": [{"mode": "hs", "kind": "dense", "values": dense_hs()}],
#        "tensor-5": [{"mode": "hs", "kind": "coords", "coords": [list(coord) for coord in sorted(HIGHLIGHTS)], "color": BLUE_HS}],
#        "tensor-6": [{"mode": "hs", "kind": "region", "base": [0, 0], "shape": [3, 2], "jumps": [1, 2], "color": BLUE_HS}],
#    },
#)
#
#tensor_viz.viz(BASE, session_bundle=bundle)
