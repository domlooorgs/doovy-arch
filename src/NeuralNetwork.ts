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
  private l1VelocityW: number[][];
  private l2VelocityW: number[][];
  private l1VelocityB: number[];
  private l2VelocityB: number[];

  private activation: 'relu' | 'swish';

  constructor(
    public inputSize: number,
    public hiddenSize: number,
    public outputSize: number,
    public learningRate: number = 0.05,
    public momentum: number = 0.9,
    public maxGrad: number = 1.0,
    activation: 'relu' | 'swish' = 'relu',
  ) {
    this.l1 = new Layer(inputSize, hiddenSize);
    this.l2 = new Layer(hiddenSize, outputSize);
    this.l1VelocityW = Array(inputSize)
      .fill(0)
      .map(() => Array(hiddenSize).fill(0));
    this.l2VelocityW = Array(hiddenSize)
      .fill(0)
      .map(() => Array(outputSize).fill(0));
    this.l1VelocityB = Array(hiddenSize).fill(0);
    this.l2VelocityB = Array(outputSize).fill(0);

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

  public applyLRDecay(factor: number = 0.99) {
    this.learningRate *= factor;
  }

  public trainStep(inputVector: number[], target: number[]): number {
    const clip = (val: number, min: number, max: number) =>
      Math.max(min, Math.min(max, val));
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
        let gradient = outputErrors[k] * hiddenOutputs[j];

        gradient = clip(gradient, -this.maxGrad, this.maxGrad);

        gradient -= 0.0001 * this.l2.weights[j][k];

        this.l2VelocityW[j][k] =
          this.momentum * this.l2VelocityW[j][k] + this.learningRate * gradient;
        this.l2.weights[j][k] += this.l2VelocityW[j][k];
      }
    }
    this.l2.bias.forEach((_, k) => {
      const gradient = outputErrors[k];
      this.l2VelocityB[k] =
        this.momentum * this.l2VelocityB[k] + this.learningRate * gradient;
      this.l2.bias[k] += this.l2VelocityB[k];
    });

    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        let gradient = hiddenErrors[j] * inputVector[i];

        gradient = clip(gradient, -this.maxGrad, this.maxGrad);

        gradient -= 0.0001 * this.l1.weights[i][j];

        this.l1VelocityW[i][j] =
          this.momentum * this.l1VelocityW[i][j] + this.learningRate * gradient;
        this.l1.weights[i][j] += this.l1VelocityW[i][j];
      }
    }
    this.l1.bias.forEach((_, j) => {
      const gradient = hiddenErrors[j];
      this.l1VelocityB[j] =
        this.momentum * this.l1VelocityB[j] + this.learningRate * gradient;
      this.l1.bias[j] += this.l1VelocityB[j];
    });

    return outputErrors.reduce((a, b) => a + b * b, 0);
  }

  public predict(inputVector: number[]): number[] {
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
      v1w: this.l1VelocityW,
      v2w: this.l2VelocityW,
      v1b: this.l1VelocityB,
      v2b: this.l2VelocityB,
    };
  }

  public loadModel(m: any): void {
    this.l1.weights = m.w1;
    this.l2.weights = m.w2;
    this.l1.bias = m.b1;
    this.l2.bias = m.b2;

    this.l1VelocityW = m.v1w;
    this.l2VelocityW = m.v2w;
    this.l1VelocityB = m.v1b;
    this.l2VelocityB = m.v2b;
  }
}
