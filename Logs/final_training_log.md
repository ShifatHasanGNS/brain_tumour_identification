Training model with text features...
Starting Brain Tumor Classification Training (use_text=True)
Memory cleared successfully

Loading datasets...
Memory cleared successfully
Found 12000 files belonging to 4 classes.
Found 800 files belonging to 4 classes.
Training samples: 300
Validation samples: 75
Test samples: 25
Memory cleared successfully

Creating new model...
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_small_minimalistic_224_1.0_float_no_top_v2.h5
[1m2128592/2128592[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 0us/step
Using MobileNetV3Small as base model
Memory cleared successfully

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)        </span>┃<span style="font-weight: bold"> Output Shape      </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Connected to      </span>┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ text_input          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ embedding           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │ <span style="color: #00af00; text-decoration-color: #00af00">12,835,456</span> │ text_input[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]  │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Embedding</span>)         │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ separable_conv1d    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">33,664</span> │ embedding[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SeparableConv1D</span>)   │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalization │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ separable_conv1d… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ max_pooling1d       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling1D</span>)      │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ image_input         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>,  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ -                 │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ max_pooling1d[<span style="color: #00af00; text-decoration-color: #00af00">0</span>]… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ MobileNetV3Small    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">576</span>) │    <span style="color: #00af00; text-decoration-color: #00af00">441,000</span> │ image_input[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ separable_conv1d_1  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">33,664</span> │ dropout_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">SeparableConv1D</span>)   │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">576</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ MobileNetV3Small… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePool…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │ separable_conv1d… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">576</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ global_average_p… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePool…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │    <span style="color: #00af00; text-decoration-color: #00af00">147,712</span> │ dropout[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ global_average_p… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ concatenate         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">384</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ dense[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>],      │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">Concatenate</span>)       │                   │            │ dropout_2[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │     <span style="color: #00af00; text-decoration-color: #00af00">98,560</span> │ concatenate[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ batch_normalizatio… │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │ dense_1[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]     │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalizatio…</span> │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)       │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │ batch_normalizat… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)         │      <span style="color: #00af00; text-decoration-color: #00af00">1,028</span> │ dropout_3[<span style="color: #00af00; text-decoration-color: #00af00">0</span>][<span style="color: #00af00; text-decoration-color: #00af00">0</span>]   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">13,593,644</span> (51.86 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">13,207,812</span> (50.38 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">385,832</span> (1.47 MB)
</pre>

Starting training...

Epoch 1/50
Epoch 1/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 228ms/step - AUC: 0.7660 - Precision: 0.5566 - Recall: 0.4421 - accuracy: 0.5163 - loss: 1.5270Epoch 1 completed in 80.51s
Estimated time remaining: 65.7 minutes
Metrics:
AUC: 0.8331
Precision: 0.6404
Recall: 0.5216
accuracy: 0.5926
loss: 1.1386
val_AUC: 0.7623
val_Precision: 0.9259
val_Recall: 0.1979
val_accuracy: 0.5263
val_loss: 1.2018
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 256ms/step - AUC: 0.7662 - Precision: 0.5569 - Recall: 0.4423 - accuracy: 0.5166 - loss: 1.5257 - val_AUC: 0.7623 - val_Precision: 0.9259 - val_Recall: 0.1979 - val_accuracy: 0.5263 - val_loss: 1.2018

Epoch 2/50
Epoch 2/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 209ms/step - AUC: 0.8919 - Precision: 0.7201 - Recall: 0.6053 - accuracy: 0.6728 - loss: 0.8384Epoch 2 completed in 69.54s
Estimated time remaining: 60.0 minutes
Metrics:
AUC: 0.9006
Precision: 0.7349
Recall: 0.6281
accuracy: 0.6885
loss: 0.8064
val_AUC: 0.7017
val_Precision: 0.5061
val_Recall: 0.4500
val_accuracy: 0.4663
val_loss: 1.6002
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m70s[0m 232ms/step - AUC: 0.8920 - Precision: 0.7202 - Recall: 0.6053 - accuracy: 0.6729 - loss: 0.8383 - val_AUC: 0.7017 - val_Precision: 0.5061 - val_Recall: 0.4500 - val_accuracy: 0.4663 - val_loss: 1.6002

Epoch 3/50
Epoch 3/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207ms/step - AUC: 0.9173 - Precision: 0.7667 - Recall: 0.6647 - accuracy: 0.7174 - loss: 0.7295Epoch 3 completed in 68.64s
Estimated time remaining: 57.1 minutes
Metrics:
AUC: 0.9221
Precision: 0.7710
Recall: 0.6732
accuracy: 0.7280
loss: 0.7082
val_AUC: 0.7695
val_Precision: 0.6443
val_Recall: 0.4717
val_accuracy: 0.5250
val_loss: 1.3194
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m69s[0m 228ms/step - AUC: 0.9173 - Precision: 0.7667 - Recall: 0.6647 - accuracy: 0.7175 - loss: 0.7294 - val_AUC: 0.7695 - val_Precision: 0.6443 - val_Recall: 0.4717 - val_accuracy: 0.5250 - val_loss: 1.3194

Epoch 4/50
Epoch 4/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208ms/step - AUC: 0.9283 - Precision: 0.7792 - Recall: 0.6948 - accuracy: 0.7401 - loss: 0.6773Epoch 4 completed in 68.97s
Estimated time remaining: 55.1 minutes
Metrics:
AUC: 0.9337
Precision: 0.7877
Recall: 0.7056
accuracy: 0.7498
loss: 0.6520
val_AUC: 0.7916
val_Precision: 0.5549
val_Recall: 0.4825
val_accuracy: 0.5142
val_loss: 1.2350
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m69s[0m 230ms/step - AUC: 0.9283 - Precision: 0.7792 - Recall: 0.6949 - accuracy: 0.7402 - loss: 0.6773 - val_AUC: 0.7916 - val_Precision: 0.5549 - val_Recall: 0.4825 - val_accuracy: 0.5142 - val_loss: 1.2350

Epoch 5/50
Epoch 5/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 208ms/step - AUC: 0.9391 - Precision: 0.7956 - Recall: 0.7155 - accuracy: 0.7560 - loss: 0.6251Epoch 5 completed in 69.21s
Estimated time remaining: 53.5 minutes
Metrics:
AUC: 0.9428
Precision: 0.8040
Recall: 0.7284
accuracy: 0.7683
loss: 0.6065
val_AUC: 0.7593
val_Precision: 0.5264
val_Recall: 0.4567
val_accuracy: 0.4850
val_loss: 1.4644
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m69s[0m 230ms/step - AUC: 0.9392 - Precision: 0.7956 - Recall: 0.7155 - accuracy: 0.7560 - loss: 0.6250 - val_AUC: 0.7593 - val_Precision: 0.5264 - val_Recall: 0.4567 - val_accuracy: 0.4850 - val_loss: 1.4644

Epoch 6/50
Epoch 6/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204ms/step - AUC: 0.9433 - Precision: 0.8077 - Recall: 0.7412 - accuracy: 0.7789 - loss: 0.6041Epoch 6 completed in 68.01s
Estimated time remaining: 51.9 minutes
Metrics:
AUC: 0.9462
Precision: 0.8118
Recall: 0.7461
accuracy: 0.7818
loss: 0.5882
val_AUC: 0.7249
val_Precision: 0.4237
val_Recall: 0.3958
val_accuracy: 0.4050
val_loss: 1.8259
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m68s[0m 226ms/step - AUC: 0.9433 - Precision: 0.8077 - Recall: 0.7412 - accuracy: 0.7789 - loss: 0.6041 - val_AUC: 0.7249 - val_Precision: 0.4237 - val_Recall: 0.3958 - val_accuracy: 0.4050 - val_loss: 1.8259

Epoch 7/50
Epoch 7/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205ms/step - AUC: 0.9460 - Precision: 0.8126 - Recall: 0.7405 - accuracy: 0.7788 - loss: 0.5881Epoch 7 completed in 68.55s
Estimated time remaining: 50.5 minutes
Metrics:
AUC: 0.9502
Precision: 0.8209
Recall: 0.7547
accuracy: 0.7921
loss: 0.5660
val_AUC: 0.8530
val_Precision: 0.6701
val_Recall: 0.5950
val_accuracy: 0.6275
val_loss: 1.0707
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m69s[0m 228ms/step - AUC: 0.9461 - Precision: 0.8126 - Recall: 0.7405 - accuracy: 0.7788 - loss: 0.5881 - val_AUC: 0.8530 - val_Precision: 0.6701 - val_Recall: 0.5950 - val_accuracy: 0.6275 - val_loss: 1.0707

Epoch 8/50
Epoch 8/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 204ms/step - AUC: 0.9538 - Precision: 0.8214 - Recall: 0.7614 - accuracy: 0.7915 - loss: 0.5446Epoch 8 completed in 67.90s
Estimated time remaining: 49.1 minutes
Metrics:
AUC: 0.9552
Precision: 0.8255
Recall: 0.7675
accuracy: 0.7987
loss: 0.5354
val_AUC: 0.8527
val_Precision: 0.6154
val_Recall: 0.5700
val_accuracy: 0.5938
val_loss: 1.1369
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m68s[0m 226ms/step - AUC: 0.9538 - Precision: 0.8214 - Recall: 0.7614 - accuracy: 0.7915 - loss: 0.5446 - val_AUC: 0.8527 - val_Precision: 0.6154 - val_Recall: 0.5700 - val_accuracy: 0.5938 - val_loss: 1.1369

Epoch 9/50
Epoch 9/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 207ms/step - AUC: 0.9546 - Precision: 0.8292 - Recall: 0.7621 - accuracy: 0.7964 - loss: 0.5397Epoch 9 completed in 69.35s
Estimated time remaining: 47.9 minutes
Metrics:
AUC: 0.9576
Precision: 0.8341
Recall: 0.7726
accuracy: 0.8054
loss: 0.5220
val_AUC: 0.8853
val_Precision: 0.7087
val_Recall: 0.6212
val_accuracy: 0.6692
val_loss: 0.9037
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m69s[0m 231ms/step - AUC: 0.9546 - Precision: 0.8292 - Recall: 0.7621 - accuracy: 0.7965 - loss: 0.5397 - val_AUC: 0.8853 - val_Precision: 0.7087 - val_Recall: 0.6212 - val_accuracy: 0.6692 - val_loss: 0.9037

Epoch 10/50
Epoch 10/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 212ms/step - AUC: 0.9572 - Precision: 0.8309 - Recall: 0.7752 - accuracy: 0.8043 - loss: 0.5244Epoch 10 completed in 70.71s
Estimated time remaining: 46.8 minutes
Metrics:
AUC: 0.9583
Precision: 0.8339
Recall: 0.7802
accuracy: 0.8087
loss: 0.5179
val_AUC: 0.9223
val_Precision: 0.7548
val_Recall: 0.6850
val_accuracy: 0.7158
val_loss: 0.7164
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m71s[0m 236ms/step - AUC: 0.9572 - Precision: 0.8309 - Recall: 0.7753 - accuracy: 0.8043 - loss: 0.5244 - val_AUC: 0.9223 - val_Precision: 0.7548 - val_Recall: 0.6850 - val_accuracy: 0.7158 - val_loss: 0.7164

Epoch 11/50
Epoch 11/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211ms/step - AUC: 0.9605 - Precision: 0.8360 - Recall: 0.7834 - accuracy: 0.8094 - loss: 0.5021Epoch 11 completed in 70.14s
Estimated time remaining: 45.6 minutes
Metrics:
AUC: 0.9621
Precision: 0.8379
Recall: 0.7893
accuracy: 0.8159
loss: 0.4923
val_AUC: 0.9129
val_Precision: 0.7586
val_Recall: 0.6692
val_accuracy: 0.7104
val_loss: 0.7709
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m70s[0m 233ms/step - AUC: 0.9605 - Precision: 0.8360 - Recall: 0.7834 - accuracy: 0.8094 - loss: 0.5021 - val_AUC: 0.9129 - val_Precision: 0.7586 - val_Recall: 0.6692 - val_accuracy: 0.7104 - val_loss: 0.7709

Epoch 12/50
Epoch 12/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 205ms/step - AUC: 0.9621 - Precision: 0.8400 - Recall: 0.7861 - accuracy: 0.8163 - loss: 0.4937Epoch 12 completed in 69.05s
Estimated time remaining: 44.4 minutes
Metrics:
AUC: 0.9640
Precision: 0.8445
Recall: 0.7967
accuracy: 0.8214
loss: 0.4811
val_AUC: 0.9313
val_Precision: 0.7797
val_Recall: 0.7271
val_accuracy: 0.7508
val_loss: 0.6817
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m69s[0m 230ms/step - AUC: 0.9621 - Precision: 0.8400 - Recall: 0.7862 - accuracy: 0.8163 - loss: 0.4936 - val_AUC: 0.9313 - val_Precision: 0.7797 - val_Recall: 0.7271 - val_accuracy: 0.7508 - val_loss: 0.6817

Epoch 13/50
Epoch 13/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 209ms/step - AUC: 0.9653 - Precision: 0.8468 - Recall: 0.7969 - accuracy: 0.8228 - loss: 0.4710Epoch 13 completed in 69.39s
Estimated time remaining: 43.2 minutes
Metrics:
AUC: 0.9666
Precision: 0.8480
Recall: 0.8031
accuracy: 0.8253
loss: 0.4630
val_AUC: 0.8691
val_Precision: 0.6702
val_Recall: 0.6383
val_accuracy: 0.6521
val_loss: 1.0945
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m69s[0m 231ms/step - AUC: 0.9653 - Precision: 0.8468 - Recall: 0.7969 - accuracy: 0.8228 - loss: 0.4710 - val_AUC: 0.8691 - val_Precision: 0.6702 - val_Recall: 0.6383 - val_accuracy: 0.6521 - val_loss: 1.0945

Epoch 14/50
Epoch 14/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211ms/step - AUC: 0.9640 - Precision: 0.8500 - Recall: 0.7994 - accuracy: 0.8270 - loss: 0.4809Epoch 14 completed in 71.12s
Estimated time remaining: 42.0 minutes
Metrics:
AUC: 0.9662
Precision: 0.8500
Recall: 0.8043
accuracy: 0.8292
loss: 0.4656
val_AUC: 0.9619
val_Precision: 0.8254
val_Recall: 0.7858
val_accuracy: 0.8042
val_loss: 0.4921
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m71s[0m 236ms/step - AUC: 0.9641 - Precision: 0.8500 - Recall: 0.7994 - accuracy: 0.8270 - loss: 0.4809 - val_AUC: 0.9619 - val_Precision: 0.8254 - val_Recall: 0.7858 - val_accuracy: 0.8042 - val_loss: 0.4921

Epoch 15/50
Epoch 15/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213ms/step - AUC: 0.9682 - Precision: 0.8514 - Recall: 0.8125 - accuracy: 0.8312 - loss: 0.4500Epoch 15 completed in 71.67s
Estimated time remaining: 40.9 minutes
Metrics:
AUC: 0.9686
Precision: 0.8551
Recall: 0.8154
accuracy: 0.8346
loss: 0.4479
val_AUC: 0.9614
val_Precision: 0.8271
val_Recall: 0.7871
val_accuracy: 0.8083
val_loss: 0.5017
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m72s[0m 239ms/step - AUC: 0.9682 - Precision: 0.8514 - Recall: 0.8125 - accuracy: 0.8313 - loss: 0.4500 - val_AUC: 0.9614 - val_Precision: 0.8271 - val_Recall: 0.7871 - val_accuracy: 0.8083 - val_loss: 0.5017

Epoch 16/50
Epoch 16/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214ms/step - AUC: 0.9687 - Precision: 0.8566 - Recall: 0.8100 - accuracy: 0.8329 - loss: 0.4478Epoch 16 completed in 71.90s
Estimated time remaining: 39.8 minutes
Metrics:
AUC: 0.9693
Precision: 0.8560
Recall: 0.8148
accuracy: 0.8359
loss: 0.4444
val_AUC: 0.9676
val_Precision: 0.8582
val_Recall: 0.7996
val_accuracy: 0.8267
val_loss: 0.4705
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m72s[0m 239ms/step - AUC: 0.9687 - Precision: 0.8566 - Recall: 0.8100 - accuracy: 0.8330 - loss: 0.4478 - val_AUC: 0.9676 - val_Precision: 0.8582 - val_Recall: 0.7996 - val_accuracy: 0.8267 - val_loss: 0.4705

Epoch 17/50
Epoch 17/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 233ms/step - AUC: 0.9735 - Precision: 0.8662 - Recall: 0.8271 - accuracy: 0.8516 - loss: 0.4117Epoch 17 completed in 77.79s
Estimated time remaining: 38.9 minutes
Metrics:
AUC: 0.9757
Precision: 0.8734
Recall: 0.8396
accuracy: 0.8589
loss: 0.3954
val_AUC: 0.9757
val_Precision: 0.8823
val_Recall: 0.8400
val_accuracy: 0.8596
val_loss: 0.4024
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m78s[0m 259ms/step - AUC: 0.9735 - Precision: 0.8662 - Recall: 0.8271 - accuracy: 0.8516 - loss: 0.4117 - val_AUC: 0.9757 - val_Precision: 0.8823 - val_Recall: 0.8400 - val_accuracy: 0.8596 - val_loss: 0.4024

Epoch 18/50
Epoch 18/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 221ms/step - AUC: 0.9749 - Precision: 0.8667 - Recall: 0.8350 - accuracy: 0.8503 - loss: 0.4005Epoch 18 completed in 73.98s
Estimated time remaining: 37.8 minutes
Metrics:
AUC: 0.9771
Precision: 0.8738
Recall: 0.8453
accuracy: 0.8595
loss: 0.3835
val_AUC: 0.9789
val_Precision: 0.8865
val_Recall: 0.8558
val_accuracy: 0.8704
val_loss: 0.3687
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 246ms/step - AUC: 0.9749 - Precision: 0.8667 - Recall: 0.8350 - accuracy: 0.8503 - loss: 0.4004 - val_AUC: 0.9789 - val_Precision: 0.8865 - val_Recall: 0.8558 - val_accuracy: 0.8704 - val_loss: 0.3687

Epoch 19/50
Epoch 19/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220ms/step - AUC: 0.9785 - Precision: 0.8810 - Recall: 0.8517 - accuracy: 0.8677 - loss: 0.3698Epoch 19 completed in 74.21s
Estimated time remaining: 36.7 minutes
Metrics:
AUC: 0.9789
Precision: 0.8824
Recall: 0.8545
accuracy: 0.8687
loss: 0.3674
val_AUC: 0.9820
val_Precision: 0.8886
val_Recall: 0.8637
val_accuracy: 0.8758
val_loss: 0.3408
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 247ms/step - AUC: 0.9785 - Precision: 0.8810 - Recall: 0.8517 - accuracy: 0.8677 - loss: 0.3698 - val_AUC: 0.9820 - val_Precision: 0.8886 - val_Recall: 0.8637 - val_accuracy: 0.8758 - val_loss: 0.3408

Epoch 20/50
Epoch 20/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 213ms/step - AUC: 0.9787 - Precision: 0.8778 - Recall: 0.8492 - accuracy: 0.8632 - loss: 0.3702Epoch 20 completed in 71.08s
Estimated time remaining: 35.5 minutes
Metrics:
AUC: 0.9790
Precision: 0.8783
Recall: 0.8499
accuracy: 0.8639
loss: 0.3671
val_AUC: 0.9780
val_Precision: 0.8842
val_Recall: 0.8492
val_accuracy: 0.8658
val_loss: 0.3786
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m71s[0m 237ms/step - AUC: 0.9787 - Precision: 0.8778 - Recall: 0.8492 - accuracy: 0.8632 - loss: 0.3702 - val_AUC: 0.9780 - val_Precision: 0.8842 - val_Recall: 0.8492 - val_accuracy: 0.8658 - val_loss: 0.3786

Epoch 21/50
Epoch 21/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 217ms/step - AUC: 0.9791 - Precision: 0.8791 - Recall: 0.8463 - accuracy: 0.8647 - loss: 0.3653Epoch 21 completed in 72.68s
Estimated time remaining: 34.4 minutes
Metrics:
AUC: 0.9802
Precision: 0.8826
Recall: 0.8523
accuracy: 0.8685
loss: 0.3574
val_AUC: 0.9832
val_Precision: 0.9019
val_Recall: 0.8737
val_accuracy: 0.8875
val_loss: 0.3295
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 242ms/step - AUC: 0.9791 - Precision: 0.8791 - Recall: 0.8463 - accuracy: 0.8647 - loss: 0.3653 - val_AUC: 0.9832 - val_Precision: 0.9019 - val_Recall: 0.8737 - val_accuracy: 0.8875 - val_loss: 0.3295

Epoch 22/50
Epoch 22/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 215ms/step - AUC: 0.9806 - Precision: 0.8805 - Recall: 0.8559 - accuracy: 0.8680 - loss: 0.3541Epoch 22 completed in 71.19s
Estimated time remaining: 33.2 minutes
Metrics:
AUC: 0.9811
Precision: 0.8869
Recall: 0.8615
accuracy: 0.8739
loss: 0.3486
val_AUC: 0.9834
val_Precision: 0.8997
val_Recall: 0.8708
val_accuracy: 0.8838
val_loss: 0.3310
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m71s[0m 237ms/step - AUC: 0.9806 - Precision: 0.8805 - Recall: 0.8560 - accuracy: 0.8680 - loss: 0.3541 - val_AUC: 0.9834 - val_Precision: 0.8997 - val_Recall: 0.8708 - val_accuracy: 0.8838 - val_loss: 0.3310

Epoch 23/50
Epoch 23/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 211ms/step - AUC: 0.9804 - Precision: 0.8886 - Recall: 0.8628 - accuracy: 0.8751 - loss: 0.3529Epoch 23 completed in 70.22s
Estimated time remaining: 32.0 minutes
Metrics:
AUC: 0.9818
Precision: 0.8917
Recall: 0.8676
accuracy: 0.8799
loss: 0.3392
val_AUC: 0.9832
val_Precision: 0.8904
val_Recall: 0.8696
val_accuracy: 0.8800
val_loss: 0.3311
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m70s[0m 234ms/step - AUC: 0.9804 - Precision: 0.8886 - Recall: 0.8628 - accuracy: 0.8751 - loss: 0.3529 - val_AUC: 0.9832 - val_Precision: 0.8904 - val_Recall: 0.8696 - val_accuracy: 0.8800 - val_loss: 0.3311

Epoch 24/50
Epoch 24/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 216ms/step - AUC: 0.9807 - Precision: 0.8867 - Recall: 0.8617 - accuracy: 0.8773 - loss: 0.3487Epoch 24 completed in 72.28s
Estimated time remaining: 30.8 minutes
Metrics:
AUC: 0.9820
Precision: 0.8898
Recall: 0.8666
accuracy: 0.8795
loss: 0.3389
val_AUC: 0.9858
val_Precision: 0.9036
val_Recall: 0.8829
val_accuracy: 0.8921
val_loss: 0.3034
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m72s[0m 241ms/step - AUC: 0.9807 - Precision: 0.8868 - Recall: 0.8617 - accuracy: 0.8773 - loss: 0.3486 - val_AUC: 0.9858 - val_Precision: 0.9036 - val_Recall: 0.8829 - val_accuracy: 0.8921 - val_loss: 0.3034

Epoch 25/50
Epoch 25/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 216ms/step - AUC: 0.9819 - Precision: 0.8966 - Recall: 0.8674 - accuracy: 0.8820 - loss: 0.3375Epoch 25 completed in 71.79s
Estimated time remaining: 29.7 minutes
Metrics:
AUC: 0.9824
Precision: 0.8971
Recall: 0.8722
accuracy: 0.8840
loss: 0.3332
val_AUC: 0.9864
val_Precision: 0.9033
val_Recall: 0.8792
val_accuracy: 0.8904
val_loss: 0.2981
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m72s[0m 239ms/step - AUC: 0.9819 - Precision: 0.8966 - Recall: 0.8674 - accuracy: 0.8821 - loss: 0.3375 - val_AUC: 0.9864 - val_Precision: 0.9033 - val_Recall: 0.8792 - val_accuracy: 0.8904 - val_loss: 0.2981

Epoch 26/50
Epoch 26/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 232ms/step - AUC: 0.9818 - Precision: 0.8865 - Recall: 0.8625 - accuracy: 0.8742 - loss: 0.3398Epoch 26 completed in 76.22s
Estimated time remaining: 28.6 minutes
Metrics:
AUC: 0.9829
Precision: 0.8899
Recall: 0.8672
accuracy: 0.8784
loss: 0.3313
val_AUC: 0.9730
val_Precision: 0.8432
val_Recall: 0.8221
val_accuracy: 0.8304
val_loss: 0.4306
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m76s[0m 254ms/step - AUC: 0.9818 - Precision: 0.8865 - Recall: 0.8625 - accuracy: 0.8742 - loss: 0.3398 - val_AUC: 0.9730 - val_Precision: 0.8432 - val_Recall: 0.8221 - val_accuracy: 0.8304 - val_loss: 0.4306

Epoch 27/50
Epoch 27/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 212ms/step - AUC: 0.9839 - Precision: 0.8894 - Recall: 0.8651 - accuracy: 0.8784 - loss: 0.3217Epoch 27 completed in 70.12s
Estimated time remaining: 27.3 minutes
Metrics:
AUC: 0.9846
Precision: 0.8948
Recall: 0.8725
accuracy: 0.8841
loss: 0.3137
val_AUC: 0.9708
val_Precision: 0.8611
val_Recall: 0.8267
val_accuracy: 0.8458
val_loss: 0.4372
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m70s[0m 233ms/step - AUC: 0.9839 - Precision: 0.8894 - Recall: 0.8651 - accuracy: 0.8785 - loss: 0.3217 - val_AUC: 0.9708 - val_Precision: 0.8611 - val_Recall: 0.8267 - val_accuracy: 0.8458 - val_loss: 0.4372

Epoch 28/50
Epoch 28/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 216ms/step - AUC: 0.9849 - Precision: 0.8995 - Recall: 0.8790 - accuracy: 0.8900 - loss: 0.3100Epoch 28 completed in 71.61s
Estimated time remaining: 26.2 minutes
Metrics:
AUC: 0.9851
Precision: 0.8985
Recall: 0.8793
accuracy: 0.8893
loss: 0.3072
val_AUC: 0.9765
val_Precision: 0.8786
val_Recall: 0.8567
val_accuracy: 0.8658
val_loss: 0.4000
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m72s[0m 238ms/step - AUC: 0.9849 - Precision: 0.8995 - Recall: 0.8790 - accuracy: 0.8900 - loss: 0.3100 - val_AUC: 0.9765 - val_Precision: 0.8786 - val_Recall: 0.8567 - val_accuracy: 0.8658 - val_loss: 0.4000

Epoch 29/50
Epoch 29/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 216ms/step - AUC: 0.9848 - Precision: 0.8987 - Recall: 0.8774 - accuracy: 0.8887 - loss: 0.3104Epoch 29 completed in 71.97s
Estimated time remaining: 25.0 minutes
Metrics:
AUC: 0.9844
Precision: 0.8968
Recall: 0.8771
accuracy: 0.8876
loss: 0.3144
val_AUC: 0.9878
val_Precision: 0.9078
val_Recall: 0.8946
val_accuracy: 0.9004
val_loss: 0.2816
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m72s[0m 240ms/step - AUC: 0.9848 - Precision: 0.8987 - Recall: 0.8774 - accuracy: 0.8887 - loss: 0.3104 - val_AUC: 0.9878 - val_Precision: 0.9078 - val_Recall: 0.8946 - val_accuracy: 0.9004 - val_loss: 0.2816

Epoch 30/50
Epoch 30/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 215ms/step - AUC: 0.9855 - Precision: 0.9018 - Recall: 0.8822 - accuracy: 0.8914 - loss: 0.3044Epoch 30 completed in 70.99s
Estimated time remaining: 23.8 minutes
Metrics:
AUC: 0.9857
Precision: 0.9015
Recall: 0.8830
accuracy: 0.8908
loss: 0.3022
val_AUC: 0.9788
val_Precision: 0.8837
val_Recall: 0.8454
val_accuracy: 0.8671
val_loss: 0.3706
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m71s[0m 236ms/step - AUC: 0.9855 - Precision: 0.9018 - Recall: 0.8822 - accuracy: 0.8914 - loss: 0.3044 - val_AUC: 0.9788 - val_Precision: 0.8837 - val_Recall: 0.8454 - val_accuracy: 0.8671 - val_loss: 0.3706

Epoch 31/50
Epoch 31/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 222ms/step - AUC: 0.9861 - Precision: 0.9037 - Recall: 0.8860 - accuracy: 0.8964 - loss: 0.2951Epoch 31 completed in 73.44s
Estimated time remaining: 22.6 minutes
Metrics:
AUC: 0.9859
Precision: 0.9043
Recall: 0.8867
accuracy: 0.8963
loss: 0.2975
val_AUC: 0.9883
val_Precision: 0.9100
val_Recall: 0.8929
val_accuracy: 0.9004
val_loss: 0.2751
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 245ms/step - AUC: 0.9861 - Precision: 0.9037 - Recall: 0.8860 - accuracy: 0.8964 - loss: 0.2951 - val_AUC: 0.9883 - val_Precision: 0.9100 - val_Recall: 0.8929 - val_accuracy: 0.9004 - val_loss: 0.2751

Epoch 32/50
Epoch 32/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 230ms/step - AUC: 0.9877 - Precision: 0.9077 - Recall: 0.8920 - accuracy: 0.8968 - loss: 0.2822Epoch 32 completed in 76.42s
Estimated time remaining: 21.5 minutes
Metrics:
AUC: 0.9870
Precision: 0.9048
Recall: 0.8906
accuracy: 0.8960
loss: 0.2877
val_AUC: 0.9887
val_Precision: 0.9137
val_Recall: 0.9004
val_accuracy: 0.9071
val_loss: 0.2669
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m76s[0m 254ms/step - AUC: 0.9877 - Precision: 0.9077 - Recall: 0.8920 - accuracy: 0.8968 - loss: 0.2822 - val_AUC: 0.9887 - val_Precision: 0.9137 - val_Recall: 0.9004 - val_accuracy: 0.9071 - val_loss: 0.2669

Epoch 33/50
Epoch 33/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 218ms/step - AUC: 0.9882 - Precision: 0.9110 - Recall: 0.8956 - accuracy: 0.9040 - loss: 0.2770Epoch 33 completed in 72.65s
Estimated time remaining: 20.3 minutes
Metrics:
AUC: 0.9891
Precision: 0.9137
Recall: 0.8989
accuracy: 0.9064
loss: 0.2663
val_AUC: 0.9896
val_Precision: 0.9173
val_Recall: 0.9054
val_accuracy: 0.9117
val_loss: 0.2605
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 242ms/step - AUC: 0.9882 - Precision: 0.9110 - Recall: 0.8957 - accuracy: 0.9040 - loss: 0.2770 - val_AUC: 0.9896 - val_Precision: 0.9173 - val_Recall: 0.9054 - val_accuracy: 0.9117 - val_loss: 0.2605

Epoch 34/50
Epoch 34/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 222ms/step - AUC: 0.9891 - Precision: 0.9177 - Recall: 0.9057 - accuracy: 0.9117 - loss: 0.2645Epoch 34 completed in 73.19s
Estimated time remaining: 19.1 minutes
Metrics:
AUC: 0.9893
Precision: 0.9189
Recall: 0.9062
accuracy: 0.9119
loss: 0.2606
val_AUC: 0.9884
val_Precision: 0.9196
val_Recall: 0.9013
val_accuracy: 0.9087
val_loss: 0.2716
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 244ms/step - AUC: 0.9891 - Precision: 0.9177 - Recall: 0.9057 - accuracy: 0.9117 - loss: 0.2645 - val_AUC: 0.9884 - val_Precision: 0.9196 - val_Recall: 0.9013 - val_accuracy: 0.9087 - val_loss: 0.2716

Epoch 35/50
Epoch 35/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220ms/step - AUC: 0.9891 - Precision: 0.9135 - Recall: 0.8991 - accuracy: 0.9072 - loss: 0.2647Epoch 35 completed in 72.77s
Estimated time remaining: 17.9 minutes
Metrics:
AUC: 0.9896
Precision: 0.9156
Recall: 0.9015
accuracy: 0.9093
loss: 0.2599
val_AUC: 0.9874
val_Precision: 0.9189
val_Recall: 0.8967
val_accuracy: 0.9079
val_loss: 0.2887
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 242ms/step - AUC: 0.9891 - Precision: 0.9135 - Recall: 0.8991 - accuracy: 0.9072 - loss: 0.2647 - val_AUC: 0.9874 - val_Precision: 0.9189 - val_Recall: 0.8967 - val_accuracy: 0.9079 - val_loss: 0.2887

Epoch 36/50
Epoch 36/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 229ms/step - AUC: 0.9898 - Precision: 0.9175 - Recall: 0.9058 - accuracy: 0.9126 - loss: 0.2534Epoch 36 completed in 75.88s
Estimated time remaining: 16.8 minutes
Metrics:
AUC: 0.9901
Precision: 0.9218
Recall: 0.9097
accuracy: 0.9160
loss: 0.2507
val_AUC: 0.9881
val_Precision: 0.9176
val_Recall: 0.9046
val_accuracy: 0.9087
val_loss: 0.2721
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m76s[0m 252ms/step - AUC: 0.9898 - Precision: 0.9175 - Recall: 0.9058 - accuracy: 0.9126 - loss: 0.2534 - val_AUC: 0.9881 - val_Precision: 0.9176 - val_Recall: 0.9046 - val_accuracy: 0.9087 - val_loss: 0.2721

Epoch 37/50
Epoch 37/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 225ms/step - AUC: 0.9904 - Precision: 0.9167 - Recall: 0.9056 - accuracy: 0.9128 - loss: 0.2489Epoch 37 completed in 74.44s
Estimated time remaining: 15.6 minutes
Metrics:
AUC: 0.9908
Precision: 0.9200
Recall: 0.9093
accuracy: 0.9157
loss: 0.2432
val_AUC: 0.9897
val_Precision: 0.9217
val_Recall: 0.9025
val_accuracy: 0.9108
val_loss: 0.2621
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 248ms/step - AUC: 0.9904 - Precision: 0.9168 - Recall: 0.9056 - accuracy: 0.9128 - loss: 0.2488 - val_AUC: 0.9897 - val_Precision: 0.9217 - val_Recall: 0.9025 - val_accuracy: 0.9108 - val_loss: 0.2621

Epoch 38/50
Epoch 38/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 228ms/step - AUC: 0.9909 - Precision: 0.9216 - Recall: 0.9094 - accuracy: 0.9152 - loss: 0.2418Epoch 38 completed in 75.17s
Estimated time remaining: 14.4 minutes
Metrics:
AUC: 0.9907
Precision: 0.9225
Recall: 0.9111
accuracy: 0.9167
loss: 0.2434
val_AUC: 0.9893
val_Precision: 0.9169
val_Recall: 0.9054
val_accuracy: 0.9100
val_loss: 0.2566
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m75s[0m 250ms/step - AUC: 0.9909 - Precision: 0.9216 - Recall: 0.9094 - accuracy: 0.9152 - loss: 0.2418 - val_AUC: 0.9893 - val_Precision: 0.9169 - val_Recall: 0.9054 - val_accuracy: 0.9100 - val_loss: 0.2566

Epoch 39/50
Epoch 39/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 221ms/step - AUC: 0.9908 - Precision: 0.9238 - Recall: 0.9136 - accuracy: 0.9175 - loss: 0.2425Epoch 39 completed in 73.57s
Estimated time remaining: 13.2 minutes
Metrics:
AUC: 0.9908
Precision: 0.9241
Recall: 0.9132
accuracy: 0.9173
loss: 0.2407
val_AUC: 0.9901
val_Precision: 0.9199
val_Recall: 0.9092
val_accuracy: 0.9150
val_loss: 0.2533
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 245ms/step - AUC: 0.9908 - Precision: 0.9238 - Recall: 0.9136 - accuracy: 0.9175 - loss: 0.2425 - val_AUC: 0.9901 - val_Precision: 0.9199 - val_Recall: 0.9092 - val_accuracy: 0.9150 - val_loss: 0.2533

Epoch 40/50
Epoch 40/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 237ms/step - AUC: 0.9907 - Precision: 0.9214 - Recall: 0.9104 - accuracy: 0.9155 - loss: 0.2453Epoch 40 completed in 78.57s
Estimated time remaining: 12.0 minutes
Metrics:
AUC: 0.9907
Precision: 0.9231
Recall: 0.9121
accuracy: 0.9177
loss: 0.2434
val_AUC: 0.9904
val_Precision: 0.9228
val_Recall: 0.9121
val_accuracy: 0.9162
val_loss: 0.2453
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m79s[0m 262ms/step - AUC: 0.9907 - Precision: 0.9214 - Recall: 0.9104 - accuracy: 0.9155 - loss: 0.2453 - val_AUC: 0.9904 - val_Precision: 0.9228 - val_Recall: 0.9121 - val_accuracy: 0.9162 - val_loss: 0.2453

Epoch 41/50
Epoch 41/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 235ms/step - AUC: 0.9921 - Precision: 0.9234 - Recall: 0.9137 - accuracy: 0.9184 - loss: 0.2271Epoch 41 completed in 77.55s
Estimated time remaining: 10.8 minutes
Metrics:
AUC: 0.9917
Precision: 0.9246
Recall: 0.9147
accuracy: 0.9198
loss: 0.2310
val_AUC: 0.9905
val_Precision: 0.9239
val_Recall: 0.9104
val_accuracy: 0.9167
val_loss: 0.2416
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m78s[0m 258ms/step - AUC: 0.9921 - Precision: 0.9234 - Recall: 0.9137 - accuracy: 0.9184 - loss: 0.2271 - val_AUC: 0.9905 - val_Precision: 0.9239 - val_Recall: 0.9104 - val_accuracy: 0.9167 - val_loss: 0.2416

Epoch 42/50
Epoch 42/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 232ms/step - AUC: 0.9914 - Precision: 0.9293 - Recall: 0.9190 - accuracy: 0.9250 - loss: 0.2326Epoch 42 completed in 77.83s
Estimated time remaining: 9.7 minutes
Metrics:
AUC: 0.9918
Precision: 0.9290
Recall: 0.9182
accuracy: 0.9234
loss: 0.2285
val_AUC: 0.9909
val_Precision: 0.9259
val_Recall: 0.9117
val_accuracy: 0.9175
val_loss: 0.2399
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m78s[0m 259ms/step - AUC: 0.9914 - Precision: 0.9293 - Recall: 0.9190 - accuracy: 0.9250 - loss: 0.2325 - val_AUC: 0.9909 - val_Precision: 0.9259 - val_Recall: 0.9117 - val_accuracy: 0.9175 - val_loss: 0.2399

Epoch 43/50
Epoch 43/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 260ms/step - AUC: 0.9913 - Precision: 0.9277 - Recall: 0.9171 - accuracy: 0.9231 - loss: 0.2338Epoch 43 completed in 86.16s
Estimated time remaining: 8.5 minutes
Metrics:
AUC: 0.9914
Precision: 0.9254
Recall: 0.9143
accuracy: 0.9205
loss: 0.2345
val_AUC: 0.9903
val_Precision: 0.9264
val_Recall: 0.9129
val_accuracy: 0.9204
val_loss: 0.2445
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m86s[0m 287ms/step - AUC: 0.9913 - Precision: 0.9277 - Recall: 0.9171 - accuracy: 0.9231 - loss: 0.2338 - val_AUC: 0.9903 - val_Precision: 0.9264 - val_Recall: 0.9129 - val_accuracy: 0.9204 - val_loss: 0.2445

Epoch 44/50
Epoch 44/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 244ms/step - AUC: 0.9918 - Precision: 0.9262 - Recall: 0.9137 - accuracy: 0.9199 - loss: 0.2248Epoch 44 completed in 80.59s
Estimated time remaining: 7.3 minutes
Metrics:
AUC: 0.9921
Precision: 0.9303
Recall: 0.9181
accuracy: 0.9242
loss: 0.2189
val_AUC: 0.9875
val_Precision: 0.9079
val_Recall: 0.8950
val_accuracy: 0.9017
val_loss: 0.2828
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 268ms/step - AUC: 0.9918 - Precision: 0.9262 - Recall: 0.9137 - accuracy: 0.9200 - loss: 0.2247 - val_AUC: 0.9875 - val_Precision: 0.9079 - val_Recall: 0.8950 - val_accuracy: 0.9017 - val_loss: 0.2828

Epoch 45/50
Epoch 45/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 232ms/step - AUC: 0.9924 - Precision: 0.9249 - Recall: 0.9125 - accuracy: 0.9179 - loss: 0.2222Epoch 45 completed in 76.87s
Estimated time remaining: 6.1 minutes
Metrics:
AUC: 0.9918
Precision: 0.9265
Recall: 0.9152
accuracy: 0.9215
loss: 0.2289
val_AUC: 0.9907
val_Precision: 0.9259
val_Recall: 0.9158
val_accuracy: 0.9187
val_loss: 0.2410
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m77s[0m 256ms/step - AUC: 0.9924 - Precision: 0.9249 - Recall: 0.9125 - accuracy: 0.9179 - loss: 0.2223 - val_AUC: 0.9907 - val_Precision: 0.9259 - val_Recall: 0.9158 - val_accuracy: 0.9187 - val_loss: 0.2410

Epoch 46/50
Epoch 46/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 244ms/step - AUC: 0.9923 - Precision: 0.9282 - Recall: 0.9183 - accuracy: 0.9225 - loss: 0.2239Epoch 46 completed in 80.97s
Estimated time remaining: 4.9 minutes
Metrics:
AUC: 0.9925
Precision: 0.9299
Recall: 0.9203
accuracy: 0.9250
loss: 0.2217
val_AUC: 0.9922
val_Precision: 0.9309
val_Recall: 0.9200
val_accuracy: 0.9250
val_loss: 0.2232
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m81s[0m 270ms/step - AUC: 0.9923 - Precision: 0.9282 - Recall: 0.9183 - accuracy: 0.9225 - loss: 0.2239 - val_AUC: 0.9922 - val_Precision: 0.9309 - val_Recall: 0.9200 - val_accuracy: 0.9250 - val_loss: 0.2232

Epoch 47/50
Epoch 47/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 238ms/step - AUC: 0.9923 - Precision: 0.9300 - Recall: 0.9212 - accuracy: 0.9261 - loss: 0.2214Epoch 47 completed in 78.68s
Estimated time remaining: 3.7 minutes
Metrics:
AUC: 0.9926
Precision: 0.9306
Recall: 0.9211
accuracy: 0.9261
loss: 0.2187
val_AUC: 0.9916
val_Precision: 0.9327
val_Recall: 0.9183
val_accuracy: 0.9275
val_loss: 0.2279
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m79s[0m 262ms/step - AUC: 0.9923 - Precision: 0.9300 - Recall: 0.9212 - accuracy: 0.9261 - loss: 0.2214 - val_AUC: 0.9916 - val_Precision: 0.9327 - val_Recall: 0.9183 - val_accuracy: 0.9275 - val_loss: 0.2279

Epoch 48/50
Epoch 48/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 236ms/step - AUC: 0.9922 - Precision: 0.9280 - Recall: 0.9173 - accuracy: 0.9234 - loss: 0.2254Epoch 48 completed in 77.84s
Estimated time remaining: 2.4 minutes
Metrics:
AUC: 0.9926
Precision: 0.9308
Recall: 0.9205
accuracy: 0.9265
loss: 0.2197
val_AUC: 0.9907
val_Precision: 0.9279
val_Recall: 0.9171
val_accuracy: 0.9233
val_loss: 0.2322
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m78s[0m 259ms/step - AUC: 0.9922 - Precision: 0.9280 - Recall: 0.9173 - accuracy: 0.9234 - loss: 0.2254 - val_AUC: 0.9907 - val_Precision: 0.9279 - val_Recall: 0.9171 - val_accuracy: 0.9233 - val_loss: 0.2322

Epoch 49/50
Epoch 49/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 250ms/step - AUC: 0.9933 - Precision: 0.9351 - Recall: 0.9247 - accuracy: 0.9288 - loss: 0.2094Epoch 49 completed in 81.81s
Estimated time remaining: 1.2 minutes
Metrics:
AUC: 0.9936
Precision: 0.9367
Recall: 0.9268
accuracy: 0.9310
loss: 0.2043
val_AUC: 0.9913
val_Precision: 0.9278
val_Recall: 0.9162
val_accuracy: 0.9208
val_loss: 0.2326
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m82s[0m 272ms/step - AUC: 0.9933 - Precision: 0.9351 - Recall: 0.9247 - accuracy: 0.9288 - loss: 0.2093 - val_AUC: 0.9913 - val_Precision: 0.9278 - val_Recall: 0.9162 - val_accuracy: 0.9208 - val_loss: 0.2326

Epoch 50/50
Epoch 50/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 231ms/step - AUC: 0.9944 - Precision: 0.9394 - Recall: 0.9287 - accuracy: 0.9352 - loss: 0.1938Epoch 50 completed in 76.37s
Estimated time remaining: 0.0 minutes
Metrics:
AUC: 0.9942
Precision: 0.9401
Recall: 0.9314
accuracy: 0.9362
loss: 0.1943
val_AUC: 0.9920
val_Precision: 0.9359
val_Recall: 0.9242
val_accuracy: 0.9312
val_loss: 0.2189
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m76s[0m 254ms/step - AUC: 0.9943 - Precision: 0.9394 - Recall: 0.9287 - accuracy: 0.9352 - loss: 0.1938 - val_AUC: 0.9920 - val_Precision: 0.9359 - val_Recall: 0.9242 - val_accuracy: 0.9312 - val_loss: 0.2189
Memory cleared successfully

Saving final model...
Final model saved to ./checkpoints/final_model_with_text.keras

Measuring model performance...
2025-03-15 19:07:30.560065: I tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence

Evaluating on test set...
[1m25/25[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 87ms/step - AUC: 0.6045 - Precision: 0.3984 - Recall: 0.3934 - accuracy: 0.3949 - loss: 4.1383
2025-03-15 19:07:35.799378: I tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence

Classification Report:

                  precision    recall  f1-score   support

    glioma_tumor     0.5667    0.1700    0.2615       200

meningioma_tumor 0.5970 0.8000 0.6838 200
pituitary_tumor 0.6860 0.2950 0.4126 200
no_tumor 0.4948 0.9550 0.6519 200

        accuracy                         0.5550       800
       macro avg     0.5861    0.5550    0.5024       800
    weighted avg     0.5861    0.5550    0.5024       800

Error computing confusion matrix: [Errno 2] No such file or directory: './benchmarks/confusion_matrix.png'

Generating plots...

Training model without text features...
Starting Brain Tumor Classification Training (use_text=False)
Memory cleared successfully

Loading datasets...
Memory cleared successfully
Found 12000 files belonging to 4 classes.
Found 800 files belonging to 4 classes.
Training samples: 300
Validation samples: 75
Test samples: 25
Memory cleared successfully

Creating new model...
Using MobileNetV3Small as base model
Memory cleared successfully

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional"</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ image_input (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ MobileNetV3Small (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">576</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">441,000</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">576</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">576</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">147,712</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │        <span style="color: #00af00; text-decoration-color: #00af00">32,896</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │           <span style="color: #00af00; text-decoration-color: #00af00">512</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">516</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">622,636</span> (2.38 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">237,828</span> (929.02 KB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">384,808</span> (1.47 MB)
</pre>

Starting training...

Epoch 1/50
Epoch 1/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 89ms/step - AUC: 0.7760 - Precision: 0.5803 - Recall: 0.4456 - accuracy: 0.5319 - loss: 1.2534Epoch 1 completed in 36.82s
Estimated time remaining: 30.1 minutes
Metrics:
AUC: 0.8351
Precision: 0.6552
Recall: 0.5169
accuracy: 0.5979
loss: 1.0443
val_AUC: 0.7500
val_Precision: 0.6632
val_Recall: 0.3758
val_accuracy: 0.5008
val_loss: 1.1903
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 114ms/step - AUC: 0.7762 - Precision: 0.5805 - Recall: 0.4459 - accuracy: 0.5322 - loss: 1.2527 - val_AUC: 0.7500 - val_Precision: 0.6632 - val_Recall: 0.3758 - val_accuracy: 0.5008 - val_loss: 1.1903

Epoch 2/50
Epoch 2/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 103ms/step - AUC: 0.9001 - Precision: 0.7394 - Recall: 0.6200 - accuracy: 0.6885 - loss: 0.7945Epoch 2 completed in 37.15s
Estimated time remaining: 29.6 minutes
Metrics:
AUC: 0.9076
Precision: 0.7526
Recall: 0.6376
accuracy: 0.7028
loss: 0.7638
val_AUC: 0.7561
val_Precision: 0.4965
val_Recall: 0.4183
val_accuracy: 0.4600
val_loss: 1.2592
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 124ms/step - AUC: 0.9001 - Precision: 0.7395 - Recall: 0.6200 - accuracy: 0.6885 - loss: 0.7944 - val_AUC: 0.7561 - val_Precision: 0.4965 - val_Recall: 0.4183 - val_accuracy: 0.4600 - val_loss: 1.2592

Epoch 3/50
Epoch 3/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 97ms/step - AUC: 0.9191 - Precision: 0.7680 - Recall: 0.6657 - accuracy: 0.7230 - loss: 0.7146Epoch 3 completed in 35.54s
Estimated time remaining: 28.6 minutes
Metrics:
AUC: 0.9227
Precision: 0.7743
Recall: 0.6764
accuracy: 0.7328
loss: 0.6986
val_AUC: 0.7492
val_Precision: 0.5210
val_Recall: 0.4542
val_accuracy: 0.4787
val_loss: 1.3488
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 119ms/step - AUC: 0.9191 - Precision: 0.7680 - Recall: 0.6658 - accuracy: 0.7230 - loss: 0.7146 - val_AUC: 0.7492 - val_Precision: 0.5210 - val_Recall: 0.4542 - val_accuracy: 0.4787 - val_loss: 1.3488

Epoch 4/50
Epoch 4/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 99ms/step - AUC: 0.9284 - Precision: 0.7857 - Recall: 0.6870 - accuracy: 0.7463 - loss: 0.6724Epoch 4 completed in 36.50s
Estimated time remaining: 28.0 minutes
Metrics:
AUC: 0.9328
Precision: 0.7896
Recall: 0.6972
accuracy: 0.7521
loss: 0.6519
val_AUC: 0.8064
val_Precision: 0.5204
val_Recall: 0.4567
val_accuracy: 0.4938
val_loss: 1.1505
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 122ms/step - AUC: 0.9284 - Precision: 0.7857 - Recall: 0.6870 - accuracy: 0.7463 - loss: 0.6724 - val_AUC: 0.8064 - val_Precision: 0.5204 - val_Recall: 0.4567 - val_accuracy: 0.4938 - val_loss: 1.1505

Epoch 5/50
Epoch 5/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 101ms/step - AUC: 0.9380 - Precision: 0.7998 - Recall: 0.7081 - accuracy: 0.7628 - loss: 0.6252Epoch 5 completed in 36.64s
Estimated time remaining: 27.4 minutes
Metrics:
AUC: 0.9390
Precision: 0.7993
Recall: 0.7142
accuracy: 0.7627
loss: 0.6202
val_AUC: 0.7980
val_Precision: 0.5803
val_Recall: 0.4938
val_accuracy: 0.5304
val_loss: 1.2401
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 122ms/step - AUC: 0.9380 - Precision: 0.7998 - Recall: 0.7081 - accuracy: 0.7628 - loss: 0.6251 - val_AUC: 0.7980 - val_Precision: 0.5803 - val_Recall: 0.4938 - val_accuracy: 0.5304 - val_loss: 1.2401

Epoch 6/50
Epoch 6/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 105ms/step - AUC: 0.9427 - Precision: 0.8104 - Recall: 0.7274 - accuracy: 0.7700 - loss: 0.6023Epoch 6 completed in 37.60s
Estimated time remaining: 26.9 minutes
Metrics:
AUC: 0.9442
Precision: 0.8105
Recall: 0.7377
accuracy: 0.7752
loss: 0.5943
val_AUC: 0.8215
val_Precision: 0.6135
val_Recall: 0.5462
val_accuracy: 0.5767
val_loss: 1.1839
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 125ms/step - AUC: 0.9427 - Precision: 0.8104 - Recall: 0.7275 - accuracy: 0.7700 - loss: 0.6023 - val_AUC: 0.8215 - val_Precision: 0.6135 - val_Recall: 0.5462 - val_accuracy: 0.5767 - val_loss: 1.1839

Epoch 7/50
Epoch 7/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 102ms/step - AUC: 0.9467 - Precision: 0.8127 - Recall: 0.7392 - accuracy: 0.7798 - loss: 0.5794Epoch 7 completed in 36.90s
Estimated time remaining: 26.3 minutes
Metrics:
AUC: 0.9488
Precision: 0.8155
Recall: 0.7468
accuracy: 0.7855
loss: 0.5687
val_AUC: 0.8415
val_Precision: 0.5937
val_Recall: 0.5437
val_accuracy: 0.5704
val_loss: 1.1296
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 123ms/step - AUC: 0.9467 - Precision: 0.8127 - Recall: 0.7392 - accuracy: 0.7798 - loss: 0.5793 - val_AUC: 0.8415 - val_Precision: 0.5937 - val_Recall: 0.5437 - val_accuracy: 0.5704 - val_loss: 1.1296

Epoch 8/50
Epoch 8/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 105ms/step - AUC: 0.9504 - Precision: 0.8152 - Recall: 0.7489 - accuracy: 0.7877 - loss: 0.5615Epoch 8 completed in 37.86s
Estimated time remaining: 25.8 minutes
Metrics:
AUC: 0.9516
Precision: 0.8194
Recall: 0.7541
accuracy: 0.7922
loss: 0.5541
val_AUC: 0.8705
val_Precision: 0.6160
val_Recall: 0.5688
val_accuracy: 0.5929
val_loss: 0.9983
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 126ms/step - AUC: 0.9504 - Precision: 0.8152 - Recall: 0.7489 - accuracy: 0.7877 - loss: 0.5614 - val_AUC: 0.8705 - val_Precision: 0.6160 - val_Recall: 0.5688 - val_accuracy: 0.5929 - val_loss: 0.9983

Epoch 9/50
Epoch 9/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 102ms/step - AUC: 0.9521 - Precision: 0.8241 - Recall: 0.7579 - accuracy: 0.7954 - loss: 0.5506Epoch 9 completed in 36.74s
Estimated time remaining: 25.2 minutes
Metrics:
AUC: 0.9542
Precision: 0.8265
Recall: 0.7641
accuracy: 0.7972
loss: 0.5387
val_AUC: 0.9153
val_Precision: 0.7424
val_Recall: 0.6712
val_accuracy: 0.6946
val_loss: 0.7490
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 122ms/step - AUC: 0.9521 - Precision: 0.8241 - Recall: 0.7579 - accuracy: 0.7954 - loss: 0.5506 - val_AUC: 0.9153 - val_Precision: 0.7424 - val_Recall: 0.6712 - val_accuracy: 0.6946 - val_loss: 0.7490

Epoch 10/50
Epoch 10/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 101ms/step - AUC: 0.9578 - Precision: 0.8286 - Recall: 0.7705 - accuracy: 0.8021 - loss: 0.5169Epoch 10 completed in 36.73s
Estimated time remaining: 24.6 minutes
Metrics:
AUC: 0.9587
Precision: 0.8353
Recall: 0.7764
accuracy: 0.8078
loss: 0.5121
val_AUC: 0.9367
val_Precision: 0.7993
val_Recall: 0.7171
val_accuracy: 0.7538
val_loss: 0.6373
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 122ms/step - AUC: 0.9578 - Precision: 0.8286 - Recall: 0.7705 - accuracy: 0.8021 - loss: 0.5169 - val_AUC: 0.9367 - val_Precision: 0.7993 - val_Recall: 0.7171 - val_accuracy: 0.7538 - val_loss: 0.6373

Epoch 11/50
Epoch 11/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 100ms/step - AUC: 0.9596 - Precision: 0.8378 - Recall: 0.7790 - accuracy: 0.8129 - loss: 0.5060Epoch 11 completed in 36.35s
Estimated time remaining: 23.9 minutes
Metrics:
AUC: 0.9601
Precision: 0.8395
Recall: 0.7828
accuracy: 0.8157
loss: 0.5024
val_AUC: 0.9496
val_Precision: 0.8111
val_Recall: 0.7479
val_accuracy: 0.7817
val_loss: 0.5692
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 121ms/step - AUC: 0.9596 - Precision: 0.8378 - Recall: 0.7790 - accuracy: 0.8129 - loss: 0.5060 - val_AUC: 0.9496 - val_Precision: 0.8111 - val_Recall: 0.7479 - val_accuracy: 0.7817 - val_loss: 0.5692

Epoch 12/50
Epoch 12/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 100ms/step - AUC: 0.9608 - Precision: 0.8375 - Recall: 0.7884 - accuracy: 0.8116 - loss: 0.4974Epoch 12 completed in 36.70s
Estimated time remaining: 23.3 minutes
Metrics:
AUC: 0.9616
Precision: 0.8419
Recall: 0.7929
accuracy: 0.8174
loss: 0.4936
val_AUC: 0.9443
val_Precision: 0.7801
val_Recall: 0.7287
val_accuracy: 0.7563
val_loss: 0.5991
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 122ms/step - AUC: 0.9608 - Precision: 0.8375 - Recall: 0.7884 - accuracy: 0.8116 - loss: 0.4974 - val_AUC: 0.9443 - val_Precision: 0.7801 - val_Recall: 0.7287 - val_accuracy: 0.7563 - val_loss: 0.5991

Epoch 13/50
Epoch 13/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 101ms/step - AUC: 0.9635 - Precision: 0.8401 - Recall: 0.7930 - accuracy: 0.8186 - loss: 0.4818Epoch 13 completed in 36.35s
Estimated time remaining: 22.7 minutes
Metrics:
AUC: 0.9634
Precision: 0.8430
Recall: 0.7952
accuracy: 0.8206
loss: 0.4814
val_AUC: 0.9362
val_Precision: 0.7465
val_Recall: 0.7029
val_accuracy: 0.7237
val_loss: 0.6563
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 121ms/step - AUC: 0.9635 - Precision: 0.8401 - Recall: 0.7930 - accuracy: 0.8186 - loss: 0.4818 - val_AUC: 0.9362 - val_Precision: 0.7465 - val_Recall: 0.7029 - val_accuracy: 0.7237 - val_loss: 0.6563

Epoch 14/50
Epoch 14/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 103ms/step - AUC: 0.9651 - Precision: 0.8454 - Recall: 0.7975 - accuracy: 0.8254 - loss: 0.4709Epoch 14 completed in 37.09s
Estimated time remaining: 22.1 minutes
Metrics:
AUC: 0.9661
Precision: 0.8500
Recall: 0.8023
accuracy: 0.8294
loss: 0.4645
val_AUC: 0.9606
val_Precision: 0.8302
val_Recall: 0.7763
val_accuracy: 0.8042
val_loss: 0.5017
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 124ms/step - AUC: 0.9651 - Precision: 0.8455 - Recall: 0.7975 - accuracy: 0.8254 - loss: 0.4708 - val_AUC: 0.9606 - val_Precision: 0.8302 - val_Recall: 0.7763 - val_accuracy: 0.8042 - val_loss: 0.5017

Epoch 15/50
Epoch 15/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 102ms/step - AUC: 0.9668 - Precision: 0.8461 - Recall: 0.8069 - accuracy: 0.8274 - loss: 0.4585Epoch 15 completed in 36.69s
Estimated time remaining: 21.5 minutes
Metrics:
AUC: 0.9676
Precision: 0.8522
Recall: 0.8110
accuracy: 0.8316
loss: 0.4533
val_AUC: 0.9708
val_Precision: 0.8637
val_Recall: 0.8158
val_accuracy: 0.8421
val_loss: 0.4367
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 122ms/step - AUC: 0.9668 - Precision: 0.8462 - Recall: 0.8069 - accuracy: 0.8274 - loss: 0.4585 - val_AUC: 0.9708 - val_Precision: 0.8637 - val_Recall: 0.8158 - val_accuracy: 0.8421 - val_loss: 0.4367

Epoch 16/50
Epoch 16/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 102ms/step - AUC: 0.9669 - Precision: 0.8487 - Recall: 0.7996 - accuracy: 0.8260 - loss: 0.4582Epoch 16 completed in 36.75s
Estimated time remaining: 20.8 minutes
Metrics:
AUC: 0.9676
Precision: 0.8517
Recall: 0.8066
accuracy: 0.8299
loss: 0.4537
val_AUC: 0.9622
val_Precision: 0.8320
val_Recall: 0.7883
val_accuracy: 0.8087
val_loss: 0.4889
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 122ms/step - AUC: 0.9670 - Precision: 0.8488 - Recall: 0.7996 - accuracy: 0.8260 - loss: 0.4582 - val_AUC: 0.9622 - val_Precision: 0.8320 - val_Recall: 0.7883 - val_accuracy: 0.8087 - val_loss: 0.4889

Epoch 17/50
Epoch 17/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 101ms/step - AUC: 0.9746 - Precision: 0.8643 - Recall: 0.8254 - accuracy: 0.8471 - loss: 0.4044Epoch 17 completed in 36.60s
Estimated time remaining: 20.2 minutes
Metrics:
AUC: 0.9747
Precision: 0.8652
Recall: 0.8290
accuracy: 0.8486
loss: 0.4025
val_AUC: 0.9768
val_Precision: 0.8826
val_Recall: 0.8454
val_accuracy: 0.8629
val_loss: 0.3896
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 122ms/step - AUC: 0.9746 - Precision: 0.8643 - Recall: 0.8254 - accuracy: 0.8472 - loss: 0.4044 - val_AUC: 0.9768 - val_Precision: 0.8826 - val_Recall: 0.8454 - val_accuracy: 0.8629 - val_loss: 0.3896

Epoch 18/50
Epoch 18/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 101ms/step - AUC: 0.9743 - Precision: 0.8675 - Recall: 0.8332 - accuracy: 0.8495 - loss: 0.4040Epoch 18 completed in 36.31s
Estimated time remaining: 19.6 minutes
Metrics:
AUC: 0.9758
Precision: 0.8699
Recall: 0.8369
accuracy: 0.8527
loss: 0.3924
val_AUC: 0.9759
val_Precision: 0.8706
val_Recall: 0.8383
val_accuracy: 0.8554
val_loss: 0.3952
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m36s[0m 121ms/step - AUC: 0.9743 - Precision: 0.8675 - Recall: 0.8332 - accuracy: 0.8496 - loss: 0.4039 - val_AUC: 0.9759 - val_Precision: 0.8706 - val_Recall: 0.8383 - val_accuracy: 0.8554 - val_loss: 0.3952

Epoch 19/50
Epoch 19/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 104ms/step - AUC: 0.9767 - Precision: 0.8756 - Recall: 0.8443 - accuracy: 0.8605 - loss: 0.3844Epoch 19 completed in 37.34s
Estimated time remaining: 19.0 minutes
Metrics:
AUC: 0.9766
Precision: 0.8757
Recall: 0.8438
accuracy: 0.8600
loss: 0.3851
val_AUC: 0.9820
val_Precision: 0.8951
val_Recall: 0.8608
val_accuracy: 0.8796
val_loss: 0.3469
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 124ms/step - AUC: 0.9767 - Precision: 0.8756 - Recall: 0.8443 - accuracy: 0.8605 - loss: 0.3844 - val_AUC: 0.9820 - val_Precision: 0.8951 - val_Recall: 0.8608 - val_accuracy: 0.8796 - val_loss: 0.3469

Epoch 20/50
Epoch 20/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 102ms/step - AUC: 0.9776 - Precision: 0.8784 - Recall: 0.8464 - accuracy: 0.8621 - loss: 0.3786Epoch 20 completed in 36.63s
Estimated time remaining: 18.4 minutes
Metrics:
AUC: 0.9786
Precision: 0.8832
Recall: 0.8528
accuracy: 0.8670
loss: 0.3689
val_AUC: 0.9824
val_Precision: 0.8880
val_Recall: 0.8558
val_accuracy: 0.8758
val_loss: 0.3392
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 122ms/step - AUC: 0.9776 - Precision: 0.8785 - Recall: 0.8464 - accuracy: 0.8621 - loss: 0.3785 - val_AUC: 0.9824 - val_Precision: 0.8880 - val_Recall: 0.8558 - val_accuracy: 0.8758 - val_loss: 0.3392

Epoch 21/50
Epoch 21/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 106ms/step - AUC: 0.9791 - Precision: 0.8826 - Recall: 0.8531 - accuracy: 0.8661 - loss: 0.3661Epoch 21 completed in 38.62s
Estimated time remaining: 17.8 minutes
Metrics:
AUC: 0.9799
Precision: 0.8848
Recall: 0.8575
accuracy: 0.8695
loss: 0.3590
val_AUC: 0.9835
val_Precision: 0.9002
val_Recall: 0.8721
val_accuracy: 0.8858
val_loss: 0.3300
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 129ms/step - AUC: 0.9791 - Precision: 0.8826 - Recall: 0.8531 - accuracy: 0.8661 - loss: 0.3661 - val_AUC: 0.9835 - val_Precision: 0.9002 - val_Recall: 0.8721 - val_accuracy: 0.8858 - val_loss: 0.3300

Epoch 22/50
Epoch 22/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 111ms/step - AUC: 0.9800 - Precision: 0.8902 - Recall: 0.8616 - accuracy: 0.8756 - loss: 0.3553Epoch 22 completed in 39.74s
Estimated time remaining: 17.3 minutes
Metrics:
AUC: 0.9802
Precision: 0.8879
Recall: 0.8607
accuracy: 0.8753
loss: 0.3531
val_AUC: 0.9831
val_Precision: 0.8873
val_Recall: 0.8629
val_accuracy: 0.8750
val_loss: 0.3303
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 133ms/step - AUC: 0.9800 - Precision: 0.8902 - Recall: 0.8616 - accuracy: 0.8756 - loss: 0.3553 - val_AUC: 0.9831 - val_Precision: 0.8873 - val_Recall: 0.8629 - val_accuracy: 0.8750 - val_loss: 0.3303

Epoch 23/50
Epoch 23/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 110ms/step - AUC: 0.9813 - Precision: 0.8842 - Recall: 0.8590 - accuracy: 0.8705 - loss: 0.3454Epoch 23 completed in 39.63s
Estimated time remaining: 16.7 minutes
Metrics:
AUC: 0.9812
Precision: 0.8869
Recall: 0.8609
accuracy: 0.8732
loss: 0.3445
val_AUC: 0.9835
val_Precision: 0.8944
val_Recall: 0.8717
val_accuracy: 0.8821
val_loss: 0.3245
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 132ms/step - AUC: 0.9813 - Precision: 0.8842 - Recall: 0.8590 - accuracy: 0.8706 - loss: 0.3454 - val_AUC: 0.9835 - val_Precision: 0.8944 - val_Recall: 0.8717 - val_accuracy: 0.8821 - val_loss: 0.3245

Epoch 24/50
Epoch 24/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 112ms/step - AUC: 0.9805 - Precision: 0.8877 - Recall: 0.8602 - accuracy: 0.8753 - loss: 0.3499Epoch 24 completed in 40.83s
Estimated time remaining: 16.1 minutes
Metrics:
AUC: 0.9810
Precision: 0.8903
Recall: 0.8649
accuracy: 0.8780
loss: 0.3455
val_AUC: 0.9856
val_Precision: 0.8995
val_Recall: 0.8767
val_accuracy: 0.8900
val_loss: 0.3047
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 136ms/step - AUC: 0.9805 - Precision: 0.8877 - Recall: 0.8602 - accuracy: 0.8753 - loss: 0.3499 - val_AUC: 0.9856 - val_Precision: 0.8995 - val_Recall: 0.8767 - val_accuracy: 0.8900 - val_loss: 0.3047

Epoch 25/50
Epoch 25/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 109ms/step - AUC: 0.9811 - Precision: 0.8857 - Recall: 0.8620 - accuracy: 0.8723 - loss: 0.3440Epoch 25 completed in 39.97s
Estimated time remaining: 15.6 minutes
Metrics:
AUC: 0.9819
Precision: 0.8912
Recall: 0.8677
accuracy: 0.8790
loss: 0.3372
val_AUC: 0.9848
val_Precision: 0.9066
val_Recall: 0.8817
val_accuracy: 0.8954
val_loss: 0.3111
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 133ms/step - AUC: 0.9811 - Precision: 0.8857 - Recall: 0.8620 - accuracy: 0.8723 - loss: 0.3440 - val_AUC: 0.9848 - val_Precision: 0.9066 - val_Recall: 0.8817 - val_accuracy: 0.8954 - val_loss: 0.3111

Epoch 26/50
Epoch 26/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 102ms/step - AUC: 0.9817 - Precision: 0.8922 - Recall: 0.8692 - accuracy: 0.8797 - loss: 0.3394Epoch 26 completed in 37.22s
Estimated time remaining: 14.9 minutes
Metrics:
AUC: 0.9825
Precision: 0.8928
Recall: 0.8705
accuracy: 0.8814
loss: 0.3316
val_AUC: 0.9854
val_Precision: 0.9042
val_Recall: 0.8846
val_accuracy: 0.8946
val_loss: 0.3019
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m37s[0m 124ms/step - AUC: 0.9817 - Precision: 0.8922 - Recall: 0.8692 - accuracy: 0.8797 - loss: 0.3394 - val_AUC: 0.9854 - val_Precision: 0.9042 - val_Recall: 0.8846 - val_accuracy: 0.8946 - val_loss: 0.3019

Epoch 27/50
Epoch 27/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 104ms/step - AUC: 0.9839 - Precision: 0.8954 - Recall: 0.8737 - accuracy: 0.8851 - loss: 0.3197Epoch 27 completed in 38.23s
Estimated time remaining: 14.3 minutes
Metrics:
AUC: 0.9834
Precision: 0.8958
Recall: 0.8767
accuracy: 0.8865
loss: 0.3231
val_AUC: 0.9863
val_Precision: 0.9069
val_Recall: 0.8888
val_accuracy: 0.8983
val_loss: 0.2932
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 127ms/step - AUC: 0.9839 - Precision: 0.8954 - Recall: 0.8738 - accuracy: 0.8851 - loss: 0.3197 - val_AUC: 0.9863 - val_Precision: 0.9069 - val_Recall: 0.8888 - val_accuracy: 0.8983 - val_loss: 0.2932

Epoch 28/50
Epoch 28/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 102ms/step - AUC: 0.9839 - Precision: 0.8997 - Recall: 0.8781 - accuracy: 0.8896 - loss: 0.3162Epoch 28 completed in 37.69s
Estimated time remaining: 13.7 minutes
Metrics:
AUC: 0.9847
Precision: 0.8997
Recall: 0.8789
accuracy: 0.8904
loss: 0.3093
val_AUC: 0.9880
val_Precision: 0.9070
val_Recall: 0.8900
val_accuracy: 0.8971
val_loss: 0.2765
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 126ms/step - AUC: 0.9839 - Precision: 0.8997 - Recall: 0.8781 - accuracy: 0.8896 - loss: 0.3162 - val_AUC: 0.9880 - val_Precision: 0.9070 - val_Recall: 0.8900 - val_accuracy: 0.8971 - val_loss: 0.2765

Epoch 29/50
Epoch 29/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 112ms/step - AUC: 0.9847 - Precision: 0.9014 - Recall: 0.8816 - accuracy: 0.8933 - loss: 0.3108Epoch 29 completed in 40.89s
Estimated time remaining: 13.1 minutes
Metrics:
AUC: 0.9844
Precision: 0.9012
Recall: 0.8831
accuracy: 0.8919
loss: 0.3122
val_AUC: 0.9871
val_Precision: 0.9157
val_Recall: 0.8921
val_accuracy: 0.9038
val_loss: 0.2853
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 136ms/step - AUC: 0.9847 - Precision: 0.9014 - Recall: 0.8816 - accuracy: 0.8933 - loss: 0.3108 - val_AUC: 0.9871 - val_Precision: 0.9157 - val_Recall: 0.8921 - val_accuracy: 0.9038 - val_loss: 0.2853

Epoch 30/50
Epoch 30/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 115ms/step - AUC: 0.9852 - Precision: 0.8979 - Recall: 0.8767 - accuracy: 0.8864 - loss: 0.3074Epoch 30 completed in 41.32s
Estimated time remaining: 12.5 minutes
Metrics:
AUC: 0.9850
Precision: 0.8990
Recall: 0.8786
accuracy: 0.8890
loss: 0.3091
val_AUC: 0.9882
val_Precision: 0.9165
val_Recall: 0.8967
val_accuracy: 0.9054
val_loss: 0.2762
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 138ms/step - AUC: 0.9852 - Precision: 0.8979 - Recall: 0.8767 - accuracy: 0.8864 - loss: 0.3074 - val_AUC: 0.9882 - val_Precision: 0.9165 - val_Recall: 0.8967 - val_accuracy: 0.9054 - val_loss: 0.2762

Epoch 31/50
Epoch 31/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 111ms/step - AUC: 0.9855 - Precision: 0.9000 - Recall: 0.8818 - accuracy: 0.8913 - loss: 0.3011Epoch 31 completed in 39.99s
Estimated time remaining: 11.9 minutes
Metrics:
AUC: 0.9852
Precision: 0.9006
Recall: 0.8833
accuracy: 0.8924
loss: 0.3033
val_AUC: 0.9863
val_Precision: 0.8987
val_Recall: 0.8800
val_accuracy: 0.8888
val_loss: 0.2931
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 133ms/step - AUC: 0.9855 - Precision: 0.9000 - Recall: 0.8818 - accuracy: 0.8913 - loss: 0.3011 - val_AUC: 0.9863 - val_Precision: 0.8987 - val_Recall: 0.8800 - val_accuracy: 0.8888 - val_loss: 0.2931

Epoch 32/50
Epoch 32/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 110ms/step - AUC: 0.9863 - Precision: 0.9018 - Recall: 0.8826 - accuracy: 0.8922 - loss: 0.2976Epoch 32 completed in 39.75s
Estimated time remaining: 11.3 minutes
Metrics:
AUC: 0.9860
Precision: 0.9054
Recall: 0.8861
accuracy: 0.8957
loss: 0.2980
val_AUC: 0.9879
val_Precision: 0.9061
val_Recall: 0.8921
val_accuracy: 0.8975
val_loss: 0.2742
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 132ms/step - AUC: 0.9862 - Precision: 0.9018 - Recall: 0.8826 - accuracy: 0.8922 - loss: 0.2976 - val_AUC: 0.9879 - val_Precision: 0.9061 - val_Recall: 0.8921 - val_accuracy: 0.8975 - val_loss: 0.2742

Epoch 33/50
Epoch 33/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 111ms/step - AUC: 0.9877 - Precision: 0.9094 - Recall: 0.8899 - accuracy: 0.9006 - loss: 0.2787Epoch 33 completed in 39.69s
Estimated time remaining: 10.7 minutes
Metrics:
AUC: 0.9885
Precision: 0.9099
Recall: 0.8934
accuracy: 0.9024
loss: 0.2717
val_AUC: 0.9884
val_Precision: 0.9099
val_Recall: 0.8958
val_accuracy: 0.9008
val_loss: 0.2677
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 132ms/step - AUC: 0.9877 - Precision: 0.9094 - Recall: 0.8900 - accuracy: 0.9006 - loss: 0.2787 - val_AUC: 0.9884 - val_Precision: 0.9099 - val_Recall: 0.8958 - val_accuracy: 0.9008 - val_loss: 0.2677

Epoch 34/50
Epoch 34/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 108ms/step - AUC: 0.9877 - Precision: 0.9101 - Recall: 0.8913 - accuracy: 0.8988 - loss: 0.2820Epoch 34 completed in 38.76s
Estimated time remaining: 10.1 minutes
Metrics:
AUC: 0.9881
Precision: 0.9115
Recall: 0.8951
accuracy: 0.9022
loss: 0.2777
val_AUC: 0.9896
val_Precision: 0.9184
val_Recall: 0.9054
val_accuracy: 0.9121
val_loss: 0.2537
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 129ms/step - AUC: 0.9877 - Precision: 0.9101 - Recall: 0.8913 - accuracy: 0.8988 - loss: 0.2820 - val_AUC: 0.9896 - val_Precision: 0.9184 - val_Recall: 0.9054 - val_accuracy: 0.9121 - val_loss: 0.2537

Epoch 35/50
Epoch 35/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 107ms/step - AUC: 0.9887 - Precision: 0.9114 - Recall: 0.8951 - accuracy: 0.9024 - loss: 0.2703Epoch 35 completed in 38.23s
Estimated time remaining: 9.5 minutes
Metrics:
AUC: 0.9892
Precision: 0.9147
Recall: 0.9001
accuracy: 0.9081
loss: 0.2633
val_AUC: 0.9895
val_Precision: 0.9190
val_Recall: 0.9029
val_accuracy: 0.9096
val_loss: 0.2577
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 127ms/step - AUC: 0.9887 - Precision: 0.9115 - Recall: 0.8951 - accuracy: 0.9024 - loss: 0.2703 - val_AUC: 0.9895 - val_Precision: 0.9190 - val_Recall: 0.9029 - val_accuracy: 0.9096 - val_loss: 0.2577

Epoch 36/50
Epoch 36/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 108ms/step - AUC: 0.9905 - Precision: 0.9220 - Recall: 0.9083 - accuracy: 0.9153 - loss: 0.2469Epoch 36 completed in 39.79s
Estimated time remaining: 8.9 minutes
Metrics:
AUC: 0.9904
Precision: 0.9218
Recall: 0.9098
accuracy: 0.9157
loss: 0.2474
val_AUC: 0.9904
val_Precision: 0.9224
val_Recall: 0.9117
val_accuracy: 0.9162
val_loss: 0.2463
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 133ms/step - AUC: 0.9905 - Precision: 0.9220 - Recall: 0.9083 - accuracy: 0.9153 - loss: 0.2469 - val_AUC: 0.9904 - val_Precision: 0.9224 - val_Recall: 0.9117 - val_accuracy: 0.9162 - val_loss: 0.2463

Epoch 37/50
Epoch 37/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 109ms/step - AUC: 0.9905 - Precision: 0.9201 - Recall: 0.9054 - accuracy: 0.9136 - loss: 0.2490Epoch 37 completed in 39.38s
Estimated time remaining: 8.2 minutes
Metrics:
AUC: 0.9897
Precision: 0.9176
Recall: 0.9047
accuracy: 0.9120
loss: 0.2575
val_AUC: 0.9890
val_Precision: 0.9122
val_Recall: 0.9008
val_accuracy: 0.9046
val_loss: 0.2649
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 131ms/step - AUC: 0.9905 - Precision: 0.9201 - Recall: 0.9054 - accuracy: 0.9136 - loss: 0.2490 - val_AUC: 0.9890 - val_Precision: 0.9122 - val_Recall: 0.9008 - val_accuracy: 0.9046 - val_loss: 0.2649

Epoch 38/50
Epoch 38/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 106ms/step - AUC: 0.9896 - Precision: 0.9201 - Recall: 0.9036 - accuracy: 0.9120 - loss: 0.2563Epoch 38 completed in 37.81s
Estimated time remaining: 7.6 minutes
Metrics:
AUC: 0.9898
Precision: 0.9210
Recall: 0.9052
accuracy: 0.9134
loss: 0.2544
val_AUC: 0.9884
val_Precision: 0.9104
val_Recall: 0.8975
val_accuracy: 0.9008
val_loss: 0.2712
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 126ms/step - AUC: 0.9896 - Precision: 0.9201 - Recall: 0.9036 - accuracy: 0.9120 - loss: 0.2563 - val_AUC: 0.9884 - val_Precision: 0.9104 - val_Recall: 0.8975 - val_accuracy: 0.9008 - val_loss: 0.2712

Epoch 39/50
Epoch 39/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 108ms/step - AUC: 0.9906 - Precision: 0.9207 - Recall: 0.9059 - accuracy: 0.9115 - loss: 0.2450Epoch 39 completed in 38.74s
Estimated time remaining: 7.0 minutes
Metrics:
AUC: 0.9903
Precision: 0.9205
Recall: 0.9068
accuracy: 0.9128
loss: 0.2485
val_AUC: 0.9908
val_Precision: 0.9265
val_Recall: 0.9142
val_accuracy: 0.9196
val_loss: 0.2395
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 129ms/step - AUC: 0.9906 - Precision: 0.9207 - Recall: 0.9059 - accuracy: 0.9115 - loss: 0.2450 - val_AUC: 0.9908 - val_Precision: 0.9265 - val_Recall: 0.9142 - val_accuracy: 0.9196 - val_loss: 0.2395

Epoch 40/50
Epoch 40/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 107ms/step - AUC: 0.9910 - Precision: 0.9206 - Recall: 0.9072 - accuracy: 0.9131 - loss: 0.2399Epoch 40 completed in 38.48s
Estimated time remaining: 6.3 minutes
Metrics:
AUC: 0.9911
Precision: 0.9224
Recall: 0.9107
accuracy: 0.9157
loss: 0.2385
val_AUC: 0.9904
val_Precision: 0.9194
val_Recall: 0.9083
val_accuracy: 0.9137
val_loss: 0.2470
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 128ms/step - AUC: 0.9910 - Precision: 0.9206 - Recall: 0.9072 - accuracy: 0.9131 - loss: 0.2399 - val_AUC: 0.9904 - val_Precision: 0.9194 - val_Recall: 0.9083 - val_accuracy: 0.9137 - val_loss: 0.2470

Epoch 41/50
Epoch 41/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 106ms/step - AUC: 0.9899 - Precision: 0.9215 - Recall: 0.9109 - accuracy: 0.9160 - loss: 0.2473Epoch 41 completed in 37.90s
Estimated time remaining: 5.7 minutes
Metrics:
AUC: 0.9907
Precision: 0.9222
Recall: 0.9114
accuracy: 0.9167
loss: 0.2412
val_AUC: 0.9904
val_Precision: 0.9184
val_Recall: 0.9100
val_accuracy: 0.9142
val_loss: 0.2461
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m38s[0m 126ms/step - AUC: 0.9899 - Precision: 0.9215 - Recall: 0.9109 - accuracy: 0.9160 - loss: 0.2473 - val_AUC: 0.9904 - val_Precision: 0.9184 - val_Recall: 0.9100 - val_accuracy: 0.9142 - val_loss: 0.2461

Epoch 42/50
Epoch 42/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 108ms/step - AUC: 0.9914 - Precision: 0.9291 - Recall: 0.9167 - accuracy: 0.9241 - loss: 0.2295Epoch 42 completed in 39.35s
Estimated time remaining: 5.1 minutes
Metrics:
AUC: 0.9914
Precision: 0.9280
Recall: 0.9147
accuracy: 0.9222
loss: 0.2310
val_AUC: 0.9892
val_Precision: 0.9157
val_Recall: 0.9004
val_accuracy: 0.9058
val_loss: 0.2582
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 131ms/step - AUC: 0.9914 - Precision: 0.9291 - Recall: 0.9167 - accuracy: 0.9241 - loss: 0.2295 - val_AUC: 0.9892 - val_Precision: 0.9157 - val_Recall: 0.9004 - val_accuracy: 0.9058 - val_loss: 0.2582

Epoch 43/50
Epoch 43/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 111ms/step - AUC: 0.9906 - Precision: 0.9254 - Recall: 0.9120 - accuracy: 0.9184 - loss: 0.2427Epoch 43 completed in 39.76s
Estimated time remaining: 4.4 minutes
Metrics:
AUC: 0.9903
Precision: 0.9230
Recall: 0.9110
accuracy: 0.9176
loss: 0.2447
val_AUC: 0.9910
val_Precision: 0.9308
val_Recall: 0.9192
val_accuracy: 0.9250
val_loss: 0.2352
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 133ms/step - AUC: 0.9906 - Precision: 0.9254 - Recall: 0.9120 - accuracy: 0.9184 - loss: 0.2427 - val_AUC: 0.9910 - val_Precision: 0.9308 - val_Recall: 0.9192 - val_accuracy: 0.9250 - val_loss: 0.2352

Epoch 44/50
Epoch 44/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 111ms/step - AUC: 0.9911 - Precision: 0.9238 - Recall: 0.9122 - accuracy: 0.9181 - loss: 0.2380Epoch 44 completed in 39.51s
Estimated time remaining: 3.8 minutes
Metrics:
AUC: 0.9917
Precision: 0.9259
Recall: 0.9144
accuracy: 0.9200
loss: 0.2314
val_AUC: 0.9902
val_Precision: 0.9255
val_Recall: 0.9117
val_accuracy: 0.9187
val_loss: 0.2456
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 132ms/step - AUC: 0.9911 - Precision: 0.9238 - Recall: 0.9122 - accuracy: 0.9182 - loss: 0.2380 - val_AUC: 0.9902 - val_Precision: 0.9255 - val_Recall: 0.9117 - val_accuracy: 0.9187 - val_loss: 0.2456

Epoch 45/50
Epoch 45/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 120ms/step - AUC: 0.9917 - Precision: 0.9301 - Recall: 0.9161 - accuracy: 0.9219 - loss: 0.2249Epoch 45 completed in 42.09s
Estimated time remaining: 3.2 minutes
Metrics:
AUC: 0.9921
Precision: 0.9302
Recall: 0.9186
accuracy: 0.9242
loss: 0.2237
val_AUC: 0.9899
val_Precision: 0.9215
val_Recall: 0.9092
val_accuracy: 0.9150
val_loss: 0.2483
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m42s[0m 140ms/step - AUC: 0.9917 - Precision: 0.9301 - Recall: 0.9161 - accuracy: 0.9219 - loss: 0.2249 - val_AUC: 0.9899 - val_Precision: 0.9215 - val_Recall: 0.9092 - val_accuracy: 0.9150 - val_loss: 0.2483

Epoch 46/50
Epoch 46/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 113ms/step - AUC: 0.9921 - Precision: 0.9301 - Recall: 0.9160 - accuracy: 0.9225 - loss: 0.2222Epoch 46 completed in 40.03s
Estimated time remaining: 2.5 minutes
Metrics:
AUC: 0.9919
Precision: 0.9312
Recall: 0.9194
accuracy: 0.9253
loss: 0.2251
val_AUC: 0.9907
val_Precision: 0.9246
val_Recall: 0.9142
val_accuracy: 0.9187
val_loss: 0.2379
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 133ms/step - AUC: 0.9921 - Precision: 0.9301 - Recall: 0.9160 - accuracy: 0.9225 - loss: 0.2222 - val_AUC: 0.9907 - val_Precision: 0.9246 - val_Recall: 0.9142 - val_accuracy: 0.9187 - val_loss: 0.2379

Epoch 47/50
Epoch 47/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 109ms/step - AUC: 0.9916 - Precision: 0.9246 - Recall: 0.9133 - accuracy: 0.9176 - loss: 0.2322Epoch 47 completed in 39.05s
Estimated time remaining: 1.9 minutes
Metrics:
AUC: 0.9920
Precision: 0.9264
Recall: 0.9165
accuracy: 0.9208
loss: 0.2276
val_AUC: 0.9914
val_Precision: 0.9294
val_Recall: 0.9162
val_accuracy: 0.9246
val_loss: 0.2287
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m39s[0m 130ms/step - AUC: 0.9916 - Precision: 0.9246 - Recall: 0.9133 - accuracy: 0.9177 - loss: 0.2322 - val_AUC: 0.9914 - val_Precision: 0.9294 - val_Recall: 0.9162 - val_accuracy: 0.9246 - val_loss: 0.2287

Epoch 48/50
Epoch 48/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 113ms/step - AUC: 0.9922 - Precision: 0.9344 - Recall: 0.9227 - accuracy: 0.9278 - loss: 0.2204Epoch 48 completed in 39.76s
Estimated time remaining: 1.3 minutes
Metrics:
AUC: 0.9922
Precision: 0.9326
Recall: 0.9218
accuracy: 0.9267
loss: 0.2209
val_AUC: 0.9909
val_Precision: 0.9224
val_Recall: 0.9117
val_accuracy: 0.9171
val_loss: 0.2397
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m40s[0m 132ms/step - AUC: 0.9922 - Precision: 0.9344 - Recall: 0.9227 - accuracy: 0.9278 - loss: 0.2204 - val_AUC: 0.9909 - val_Precision: 0.9224 - val_Recall: 0.9117 - val_accuracy: 0.9171 - val_loss: 0.2397

Epoch 49/50
Epoch 49/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 115ms/step - AUC: 0.9937 - Precision: 0.9354 - Recall: 0.9266 - accuracy: 0.9298 - loss: 0.2052Epoch 49 completed in 41.01s
Estimated time remaining: 0.6 minutes
Metrics:
AUC: 0.9935
Precision: 0.9361
Recall: 0.9271
accuracy: 0.9308
loss: 0.2070
val_AUC: 0.9918
val_Precision: 0.9318
val_Recall: 0.9217
val_accuracy: 0.9271
val_loss: 0.2259
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m41s[0m 137ms/step - AUC: 0.9937 - Precision: 0.9354 - Recall: 0.9266 - accuracy: 0.9298 - loss: 0.2052 - val_AUC: 0.9918 - val_Precision: 0.9318 - val_Recall: 0.9217 - val_accuracy: 0.9271 - val_loss: 0.2259

Epoch 50/50
Epoch 50/50
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 125ms/step - AUC: 0.9925 - Precision: 0.9344 - Recall: 0.9259 - accuracy: 0.9310 - loss: 0.2166Epoch 50 completed in 43.78s
Estimated time remaining: 0.0 minutes
Metrics:
AUC: 0.9936
Precision: 0.9383
Recall: 0.9310
accuracy: 0.9348
loss: 0.2032
val_AUC: 0.9913
val_Precision: 0.9304
val_Recall: 0.9192
val_accuracy: 0.9250
val_loss: 0.2286
[1m300/300[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m44s[0m 146ms/step - AUC: 0.9925 - Precision: 0.9344 - Recall: 0.9259 - accuracy: 0.9310 - loss: 0.2166 - val_AUC: 0.9913 - val_Precision: 0.9304 - val_Recall: 0.9192 - val_accuracy: 0.9250 - val_loss: 0.2286
Memory cleared successfully

Saving final model...
Final model saved to ./checkpoints/final_model_without_text.keras

Measuring model performance...

Evaluating on test set...
[1m25/25[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 83ms/step - AUC: 0.5943 - Precision: 0.3986 - Recall: 0.3940 - accuracy: 0.4001 - loss: 4.0014
2025-03-15 19:39:51.320468: I tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence

Classification Report:

                  precision    recall  f1-score   support

    glioma_tumor     0.5926    0.1600    0.2520       200

meningioma_tumor 0.6142 0.8200 0.7024 200
pituitary_tumor 0.7614 0.3350 0.4653 200
no_tumor 0.4910 0.9600 0.6497 200

        accuracy                         0.5687       800
       macro avg     0.6148    0.5687    0.5173       800
    weighted avg     0.6148    0.5687    0.5173       800

Generating plots...
No artists with labels found to put in legend. Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
No artists with labels found to put in legend. Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
No artists with labels found to put in legend. Note that artists whose label start with an underscore are ignored when legend() is called with no argument.

## Model Comparison Summary:

## Metric With Text Without Text

Accuracy 0.5550 0.5688  
Precision 0.5611 0.5685  
Recall 0.5512 0.5600  
AUC 0.7488 0.7414  
Training Time (s) 3682.1650 1922.9789  
Inference Time (s) 0.1329 0.1328  
Model Size (MB) 153.0672 4.5914  
Memory Usage (MB) 936.8125 617.5938

---

Comparison plots saved to: ./benchmarks

![confusion_matrix](./benchmarks/confusion_matrix.png)
