/**
 * Copyright 2026 SoTeen Studio
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

import { NeuralNetwork } from './NeuralNetwork.js';
import * as Act from './activations/relu.js';

export class Generative extends NeuralNetwork {
  private weightsEncoder: number[][];

  constructor(
    public inputSize: number,
    public hiddenSize: number,
    public latentSize: number,
    public learningRate: number = 0.05,
  ) {
    super(inputSize, hiddenSize, inputSize, learningRate);

    this.weightsEncoder = Array.from({ length: inputSize }, () =>
      Array.from(
        { length: latentSize * 2 },
        () => (Math.random() - 0.5) * Math.sqrt(2 / inputSize),
      ),
    );
  }

  public encode(inputVector: number[]): { mu: number[]; logVar: number[] } {
    const encoderOutput = new Array(this.latentSize * 2).fill(0);
    for (let j = 0; j < this.latentSize * 2; j++) {
      let sum = 0;
      for (let i = 0; i < this.inputSize; i++) {
        sum += inputVector[i] * this.weightsEncoder[i][j];
      }

      encoderOutput[j] = Act.relu(sum);
    }
    return {
      mu: encoderOutput.slice(0, this.latentSize),
      logVar: encoderOutput.slice(this.latentSize),
    };
  }

  public trainVAE(inputVector: number[]): number {
    const { mu, logVar } = this.encode(inputVector);
    const z = this.reparameterize(mu, logVar);
    const reconstruction = this.generate(z);

    let reconLoss = 0;
    const encoderGradients = Array.from({ length: this.inputSize }, () =>
      new Array(this.latentSize * 2).fill(0),
    );

    for (let i = 0; i < this.inputSize; i++) {
      const val = Math.max(0.001, Math.min(0.999, reconstruction[i]));
      reconLoss -=
        inputVector[i] * Math.log(val) +
        (1 - inputVector[i]) * Math.log(1 - val);

      const delta = (reconstruction[i] - inputVector[i]) * this.learningRate;
      for (let j = 0; j < this.latentSize * 2; j++) {
        this.weightsEncoder[i][j] -= delta * 0.1;
      }
    }

    const beta = 0.01;
    const klDivergence =
      -0.5 *
      beta *
      mu.reduce(
        (acc, m, i) =>
          acc + (1 + logVar[i] - Math.pow(m, 2) - Math.exp(logVar[i])),
        0,
      );

    return reconLoss + klDivergence;
  }

  public predict(inputVector: number[]): number[] {
    const { mu, logVar } = this.encode(inputVector);

    const z = mu;

    return this.generate(z);
  }
}
