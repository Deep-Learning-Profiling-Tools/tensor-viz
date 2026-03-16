import numpy as np
from tensor_viz import viz
import tensor_viz
from tensor_viz.bundle import create_session_data

# TODO: allow letter+ N non-letters dims (e.g. T0T1T2)
# TODO: advanced: add checkbox for morton vs contiguous dimension mapping
# TODO: advanced: show slices same place
# TODO: check embeddability

DEMO = 1

if DEMO == 0: # basic
    x = np.random.randn(2,3)
    viz(x)
if DEMO == 1: # linear layout
    x = (
        np.arange(2048)
        .reshape(2,2,2,2,2,2,2,2,2,2,2)
        #.transpose(7, 0, 1, 9, 2, 3, 4, 8, 5, 6, 10)
        #.reshape(128, 16)
        #.reshape(16, 8, 16)
    )
    #import matplotlib.pyplot as plt
    #plt.imshow(x)
    #plt.show()
    viz(x, labels="W0 W1 T0 T1 T2 T3 T4 R0 R1 R2 R3")
if DEMO == 2: # multi-tensor view
    SHAPE = (3, 4)
    BASE = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    BLUE_RGBA = [0, 90, 255, 255]
    BLUE_HS = [240, 1]
    HIGHLIGHTS = {(0, 1), (1, 2), (2, 0)}


    def dense_rgba() -> list[float]:
        values: list[float] = []
        for row in range(SHAPE[0]):
            for col in range(SHAPE[1]):
                values.extend(BLUE_RGBA if (row, col) in HIGHLIGHTS else [144, 164, 174, 255])
        return values


    def dense_hs() -> list[float]:
        values: list[float] = []
        for row in range(SHAPE[0]):
            for col in range(SHAPE[1]):
                values.extend(BLUE_HS if (row, col) in HIGHLIGHTS else [0, 0])
        return values


    tensors = {
        "Dense RGBA": BASE,
        "Coords RGBA": BASE + 100,
        "Region RGBA": BASE + 200,
        "Dense HS": BASE + 300,
        "Coords HS": BASE + 400,
        "Region HS": BASE + 500,
    }

    session_data = create_session_data(
        tensors,
        color_instructions={
            "tensor-1": [{"mode": "rgba", "kind": "dense", "values": dense_rgba()}],
            "tensor-2": [{"mode": "rgba", "kind": "coords", "coords": [list(coord) for coord in sorted(HIGHLIGHTS)], "color": BLUE_RGBA}],
            "tensor-3": [{"mode": "rgba", "kind": "region", "base": [0, 0], "shape": [3, 2], "jumps": [1, 2], "color": BLUE_RGBA}],
            "tensor-4": [{"mode": "hs", "kind": "dense", "values": dense_hs()}],
            "tensor-5": [{"mode": "hs", "kind": "coords", "coords": [list(coord) for coord in sorted(HIGHLIGHTS)], "color": BLUE_HS}],
            "tensor-6": [{"mode": "hs", "kind": "region", "base": [0, 0], "shape": [3, 2], "jumps": [1, 2], "color": BLUE_HS}],
        },
    )

    viz(BASE, session_data=session_data)
if DEMO == 3: # big boy
    #x = np.random.randn(4096,4096)
    x = np.random.randn(1024,1024)
    #x = np.random.randn(*[4]*10)
    viz(x)
if DEMO == 4: # tabs
    t1 = tensor_viz.Tab("t1")
    x = np.random.randn(2,2)
    t1.viz(x)

    t2 = tensor_viz.Tab("t2")
    x = np.random.randn(2,2,2,2,2,2,2,2,2)
    t2.viz(x)
    viz([t1, t2])
if DEMO == 5: # plt
    import time
    from PIL import Image
        
    #import matplotlib.pyplot as plt
    #plt.imshow(np.array(Image.open('/home/trtx/Projects/deer1024.jpg')))
    #plt.show()


    #viz(np.array(Image.open('/home/trtx/Projects/deer1024.jpg')), labels="HWC")
    viz(np.array(Image.open('/home/trtx/Projects/turtle32.png')), labels="HWC")
