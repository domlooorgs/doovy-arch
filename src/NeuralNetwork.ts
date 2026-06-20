/**
 * Copyright 2026 SoTeen Studio
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

import { Layer } from './Layer.js';
import * as Act from './activations/index.js';

export class NeuralNetwork {
  private l1: Layer;
  private l2: Layer;

  private activation: 'relu' | 'swish';

  constructor(
    public inputSize: number,
    public hiddenSize: number,
    public outputSize: number,
    public learningRate: number = 0.05,
    activation: 'relu' | 'swish' = 'relu',
  ) {
    this.l1 = new Layer(inputSize, hiddenSize);
    this.l2 = new Layer(hiddenSize, outputSize);
    this.activation = activation;
  }

  private get activationFunc() {
    return this.activation === 'swish' ? Act.swish : Act.relu;
  }
  private get derivativeFunc() {
    return this.activation === 'swish'
      ? Act.swishDerivative
      : Act.reluDerivative;
  }

  public trainStep(inputVector: number[], target: number[]): number {
    const hiddenOutputs = this.l1.bias.map((b, j) =>
      this.activationFunc(
        b +
          inputVector.reduce((sum, x, i) => sum + x * this.l1.weights[i][j], 0),
      ),
    );

    const logits = this.l2.bias.map(
      (b, k) =>
        b +
        hiddenOutputs.reduce((sum, h, j) => sum + h * this.l2.weights[j][k], 0),
    );

    const finalOutputs = Act.softmax(logits);
    const outputErrors = target.map((t, k) => t - finalOutputs[k]);

    const hiddenErrors = hiddenOutputs.map(
      (h, j) =>
        this.l2.weights[j].reduce((err, w, k) => err + outputErrors[k] * w, 0) *
        this.derivativeFunc(h),
    );

    for (let j = 0; j < this.hiddenSize; j++) {
      for (let k = 0; k < this.outputSize; k++) {
        this.l2.weights[j][k] +=
          this.learningRate * outputErrors[k] * hiddenOutputs[j];
      }
    }
    this.l2.bias.forEach(
      (_, k) => (this.l2.bias[k] += this.learningRate * outputErrors[k]),
    );

    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.l1.weights[i][j] +=
          this.learningRate * hiddenErrors[j] * inputVector[i];
      }
    }
    this.l1.bias.forEach(
      (_, j) => (this.l1.bias[j] += this.learningRate * hiddenErrors[j]),
    );

    return outputErrors.reduce((a, b) => a + b * b, 0);
  }

  public predict(inputVector: number[]): number[] {
    // Gunakan getter activationFunc supaya konsisten dengan pilihan user
    const hiddenOutputs = this.l1.bias.map((b, j) =>
      this.activationFunc(
        b + inputVector.reduce((sum, x, i) => sum + x * this.l1.weights[i][j], 0),
      ),
    );

    const logits = this.l2.bias.map(
      (b, k) =>
        b +
        hiddenOutputs.reduce((sum, h, j) => sum + h * this.l2.weights[j][k], 0),
    );

    return Act.softmax(logits);
  }

  public generate(latentVector: number[]): number[] {
    const hidden = this.l1.bias.map((b, j) =>
      Act.relu(
        b +
          latentVector.reduce(
            (sum, x, i) => sum + x * this.l1.weights[i][j],
            0,
          ),
      ),
    );

    return this.l2.bias.map(
      (b, k) =>
        1 /
        (1 +
          Math.exp(
            -(
              b +
              hidden.reduce((sum, h, j) => sum + h * this.l2.weights[j][k], 0)
            ),
          )),
    );
  }

  public reparameterize(mu: number[], logVar: number[]): number[] {
    return mu.map((m, i) => {
      const std = Math.exp(0.5 * logVar[i]);
      const epsilon = this.sampleNormal();
      return m + std * epsilon;
    });
  }

  private sampleNormal(): number {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  public getModel(): any {
    return {
      w1: this.l1.weights,
      w2: this.l2.weights,
      b1: this.l1.bias,
      b2: this.l2.bias,
    };
  }

  public loadModel(m: any): void {
    this.l1.weights = m.w1;
    this.l2.weights = m.w2;
    this.l1.bias = m.b1;
    this.l2.bias = m.b2;
  }
}
