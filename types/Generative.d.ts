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
export declare class Generative extends NeuralNetwork {
    inputSize: number;
    hiddenSize: number;
    latentSize: number;
    learningRate: number;
    private weightsEncoder;
    constructor(inputSize: number, hiddenSize: number, latentSize: number, learningRate?: number);
    encode(inputVector: number[]): {
        mu: number[];
        logVar: number[];
    };
    trainVAE(inputVector: number[]): number;
    predict(inputVector: number[]): number[];
}
