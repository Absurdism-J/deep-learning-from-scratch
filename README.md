# Deep Learning from Scratch

Hand-typed implementations while working through 《深度学习入门》(斋藤康毅).
Building neural networks with pure NumPy — no frameworks, no shortcuts.

## Progress

- [x] Ch1: Python & NumPy basics
- [x] Ch2: Perceptron — AND / NAND / OR, and XOR by composition
- [x] Ch3: Neural network (forward propagation) — MNIST inference 93.52% (pretrained weights)
- [x] Ch4: Loss functions & numerical gradient
- [x] Ch5: Backpropagation from scratch — **self-trained MNIST 97.08%** (test set, 10k images)
- [x] MNIST recognition ≥ 90% accuracy (self-trained full pipeline) — 97.08%

## Results

Two numbers, one distinction:

| Run | Accuracy | Weights |
|---|---|---|
| Ch3 inference | 93.52% | book's pretrained `sample_weight.pkl` |
| Ch5 self-trained | **97.08%** | every byte learned by hand-written backprop |

Network: 784-50-10, sigmoid hidden layer, softmax + cross-entropy output.
Trained with mini-batch SGD (lr=0.1, batch=100), 10000 iterations ≈ 17 epochs.
`ch05/train_neuralnet_update.py` reproduces the full run.

## Structure

    ch01/   Python & NumPy basics
    ch02/   Perceptron gates (AND, NAND, OR) and XOR composed from them
    ch03/   step_function.py, sigmoid.py, RELU.py — activation functions
            three_layer_neural_network.py — 3-layer forward propagation
            neuralnet_mnist.py — MNIST inference, 93.52% on 10k test images
            mnist_show.py — visualize a raw MNIST image
            softmax_native.py / softmax_fixed.py — naive vs overflow-safe softmax
    ch04/   mse.py, cross_entropy_error.py — loss functions (one-hot & label versions)
            numerical_diff.py — numerical differentiation & gradient
            gradient_descent.py — gradient descent optimizer
            simple_net.py — gradient w.r.t. weight matrix (lambda adapter)
            two_layer_net.py — TwoLayerNet trained via numerical gradient (the slow path)
            overfit_test.py — single-batch overfit health check
            train_neuralnet.py — numerical-gradient training loop
    ch05/   mul_layer.py, add_layer.py — computational graph building blocks
            buy_apple_orange.py — AddLayer / MulLayer with the shopping example
            relu_layer.py, sigmoid_layer.py — activation layers
            affine_batch.py — Affine layer: broadcast forward, sum backward
            softmax_with_loss.py — SoftmaxWithLoss layer
            two_layer_net_update.py — TwoLayerNet with backprop gradient
            gradient_check.py — numerical vs backprop gradient agreement check
            train_neuralnet_update.py — full training run → 97.08% test accuracy
    dataset/  vendored MNIST loader (from the book's official repo)

## How to Run

    python ch02/perceptron_one.py   # each gate file runs its own truth-table self-test
    python ch02/perceptron-XOR.py   # XOR built from the three gates above
    python ch03/sigmoid.py          # activation functions with self-test
    python ch03/neuralnet_mnist.py  # MNIST inference with pretrained weights (93.52%)
    python ch05/buy_apple_orange.py # forward + backward through a shopping graph
    python ch05/gradient_check.py   # verify backprop against numerical gradient
    python ch05/train_neuralnet_update.py  # full training run (a few minutes)

## Highlights

**XOR — the first "deep" moment.** A single perceptron can only draw one
straight line, so it can never separate XOR's four points. The fix is
composition: `XOR = AND(OR(x), NAND(x))` — three self-written gates wired
in two layers. What one layer cannot do, stacked layers can.

**Nonlinearity is the point.** Stacking linear layers still gives a linear
map — depth without nonlinear activation is an illusion. Step → sigmoid →
ReLU is the story of making activation both nonlinear and trainable.

**Accuracy as a diagnostic instrument.** A 3% accuracy drop (90.43% vs the
expected ~93.5%) exposed a one-letter bug — feeding pre-activation `a2`
instead of post-sigmoid `z2` into the output layer. The accuracy delta
measured the exact "weight" of one sigmoid layer.

**Gradient check asks a different question.** After all wiring passed
inspection, gradient check still flagged a clean 100× mismatch. Both sides
were computing correctly — numerical gradient probed the real function,
backprop probed the intended one, and a leftover calibration hack
(`softmax(x*0.01)`) made them two different functions. Gradient check
doesn't ask "who is wrong"; it asks "is the function in your head the same
as the function in the code". Initial loss ≈ ln(10) comes free from proper
initialization — no calibration tape needed.

**loss = ln(10) + ln(50): the constant that confessed.** A loss frozen at
6.2 decoded as ln(10)+ln(50): a 1-D softmax summing over the whole (50,10)
batch instead of per row. Inference stayed correct (argmax is
order-invariant) — the bug hid until it met the loss. A number that refuses
to move is also a measurement.

**Why backpropagation exists.** This network has 39,760 parameters.
Numerical gradient costs ~2 loss evaluations per parameter — ~80,000
forward passes per iteration. Backprop costs one forward + one backward.
The speed gap (~2–3×10⁴×) is not convenience; it is the difference between
"five minutes" and "never finishes".

## Notes — pitfalls logged

- **Margin matters**: a decision boundary that sits exactly on data points
  misclassifies them. Good parameters keep distance from the data (→ SVM).
- **Module side effects**: top-level code runs on import. Every module
  guards its self-test with `if __name__ == "__main__":` and passes its own
  truth table before joining a network. Second bite logged (10-03):
  PyCharm's autocomplete imported an old training script, whose top-level
  loop executed on import and whose output impersonated the test result.
- Module names must be valid Python identifiers — `perceptron-one.py` cannot
  be imported; use underscores.
- **Forever-false conditions**: `i % 10 == 0 and i % 10 == 1` never fires.
  A print that never prints hides progress, not just clutter.
- **The silent dtype trap.** Assigning floats into an int array truncates
  without any error (3.0001 → 3). Inside numerical gradient this silently
  corrupted the step size and produced absurd gradients (25000 instead of 6).
  Two truncation points identified: the input array mid-computation, and the
  output array at storage.
- **Sync discipline**: edit locally, push from local. If the web editor is
  ever used, `git pull` before the next push.
