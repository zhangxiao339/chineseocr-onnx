# chineseocr-onnx
chineres ocr from picture, 中英文本检测与文字识别，dense-ctc，dbnet，crnn，pse，unet等模型

# run
* set the image file path in main.py
* python main.py

# train
* text rec: https://github.com/zhangxiao339/DenseCTC-Keras
* dbnet: coming soon
* pse: todo
* unet: comming soon

# demo
* online demo: http://119.29.166.38:8787/ocr?type=print_ocr
* data/test/demo.png
![image](./data/test/demo.png)
* result
    > line: 1, A.上升过程中克服重力做的功大于下降过程中重力做的功, 0.9922532439231873<br>
    line: 2, B.上升过程中克服重力做的功等于下降过程中重力做的功, 0.9885271787643433<br>
    line: 3, C.上升过程中克服重力做功的平均功率大于下降过程中重力, 0.991519570350647<br>
    line: 4, 做故功的平均功率, 0.9811575412750244<br>
    line: 5, D.上升过程中克服重力做功的平均功率等于下降过程中重力, 0.9894487261772156<br>
    line: 6, 做功的平均功率, 0.9874854683876038<br>
* total time on cpu(mac book): 1.12s
