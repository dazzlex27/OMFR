# %%

import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('t1')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)

# %%

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Method-1, use FaceAnalysis
app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))

img = ins_get_image('t1')
# Method-2, load model directly
detector = insightface.model_zoo.get_model('det_10g.onnx')
detector.prepare(ctx_id=0, det_size=(640, 640))
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("output.jpg", rimg)

# %%


# import cv2
# import numpy as np
# import insightface
# from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image

# handler = insightface.model_zoo.get_model('arcface50.onnx')
# handler.prepare(ctx_id=0)
# # %%

# %%
