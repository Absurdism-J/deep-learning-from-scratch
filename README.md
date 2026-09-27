# Deep Learning from Scratch

Hand-typed implementations while working through 《深度学习入门》(斋藤康毅).
Building neural networks with pure NumPy — no frameworks, no shortcuts.

## Progress

- [x] Ch1: Python & NumPy basics
- [x] Ch2: Perceptron — AND / NAND / OR, and XOR by composition
- [x] Ch3: Neural network (forward propagation) — MNIST inference 93.52%
- [ ] Ch4: Loss functions & gradient (in progress)
- [ ] Ch5: Backpropagation from scratch
- [ ] MNIST recognition ≥ 90% accuracy (self-trained full pipeline)

## Structure

    ch01/   Python & NumPy basics
    ch02/   Perceptron gates (AND, NAND, OR) and XOR composed from them
    ch03/   step_function.py, sigmoid.py, RELU.py — activation functions
            three_layer_neural_network.py — 3-layer forward propagation
            neuralnet_mnist.py — MNIST inference, 93.52% on 10k test images
            mnist_show.py — visualize a raw MNIST image
            softmax_native.py / softmax_fixed.py — naive vs overflow-safe softmax
    ch04/   MSE.py, Cross_Entropy_Error.py — loss functions (one-hot & label versions)
            numerical_diff.py — numerical differentiation & gradient (loop + closure versions)
            gradient_descent.py — gradient descent optimizer
            simple_net.py — gradient w.r.t. weight matrix (WIP)
    dataset/  vendored MNIST loader (from the book's official repo)

## How to Run

    python ch02/perceptron_one.py   # each gate file runs its own truth-table self-test
    python ch02/perceptron-XOR.py   # XOR built from the three gates above
    python ch03/sigmoid.py          # activation functions with self-test
    python ch03/neuralnet_mnist.py  # MNIST inference with pretrained weights (93.52%)

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

**The silent dtype trap.** Assigning floats into an int array truncates
without any error (3.0001 → 3). Inside numerical gradient this silently
corrupted the step size and produced absurd gradients (25000 instead of 6).
Two truncation points identified: the input array mid-computation, and the
output array at storage.

## Notes — pitfalls logged

- **Margin matters**: a decision boundary that sits exactly on data points
  misclassifies them. Good parameters keep distance from the data (→ SVM).
- **Module side effects**: top-level test code runs on import. Every module
  guards its self-test with `if __name__ == "__main__":` and passes its own
  truth table before joining a network.
- Module names must be valid Python identifiers — `perceptron-one.py` cannot
  be imported; use underscores.
- **Sync discipline**: edit locally, push from local. If the web editor is
  ever used, `git pull` before the next push.