## Experimental Report

### Graphs

<div style="text-align: center;">
  <figure style="display: inline-block; margin-right: 20px;">
    <img src="/Users/yuxuan/Library/Application Support/typora-user-images/image-20240128152121557.png" style="width: 300px; height: auto;">
    <figcaption>Comparison of Loss in each Epoch</figcaption>
  </figure>
  <figure style="display: inline-block;">
    <img src="/Users/yuxuan/Library/Application Support/typora-user-images/image-20240128152139264.png" style="width: 300px; height: auto;">
    <figcaption>Comparison of Accuracy in each Epoch</figcaption>
  </figure>
</div>


### Comparison between Different Momentums

| Momentum | Aver. Training Acc | Aver. Validation Acc | Final Test Acc |
| -------- | ------------------ | -------------------- | -------------- |
| 0        | 0.7324             | 0.7718               | 0.7352         |
| 0.9      | 0.7304             | 0.7710               | 0.7383         |

### Comparison between Different Learning Rates

| Learning Rate | Aver. Training Acc | Aver. Validation Acc | Final Test Acc |
| ------------- | ------------------ | -------------------- | -------------- |
| 0.001         | 0.7013             | 0.7416               | 0.7090         |
| 0.01          | 0.7324             | 0.7718               | 0.7352         |
| 0.05          | 0.7301             | 0.7706               | 0.7378         |

### Comparison between Different Batch Sizes

| Batch Size | Aver. Training Acc | Aver. Validation Acc | Final Test Acc |
| ---------- | ------------------ | -------------------- | -------------- |
| 50         | 0.7308             | 0.7714               | 0.7386         |
| 100        | 0.7324             | 0.7718               | 0.7352         |
| 200        | 0.7295             | 0.7688               | 0.7355         |

### Conclusion

From the data listed above, we can see that

- Using Momentum can somehow improve the performance.
- Low Learning Rate may have worse performance.
- It is possible to improve performance regardless of whether the Batch Size is small or large.
