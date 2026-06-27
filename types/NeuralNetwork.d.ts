/**
 * Copyright 2026 SoTeen Studio
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
export declare class NeuralNetwork {
    inputSize: number;
    hiddenSize: number;
    outputSize: number;
    learningRate: number;
    momentum: number;
    maxGrad: number;
    private l1;
    private l2;
    private l1VelocityW;
    private l2VelocityW;
    private l1VelocityB;
    private l2VelocityB;
    private activation;
    constructor(inputSize: number, hiddenSize: number, outputSize: number, learningRate?: number, momentum?: number, maxGrad?: number, activation?: 'relu' | 'swish');
    private get activationFunc();
    private get derivativeFunc();
    applyLRDecay(factor?: number): void;
    trainStep(inputVector: number[], target: number[]): number;
    predict(inputVector: number[]): number[];
    generate(latentVector: number[]): number[];
    reparameterize(mu: number[], logVar: number[]): number[];
    private sampleNormal;
    getModel(): any;
    loadModel(m: any): void;
}
