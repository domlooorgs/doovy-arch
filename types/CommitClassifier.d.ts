/**
 * Copyright 2026 SoTeen Studio
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
export declare const CATEGORIES: readonly ["FEATURES", "BUG_FIXES", "MAINTENANCE", "UNCATEGORIZED"];
type Category = (typeof CATEGORIES)[number];
export interface CommitClassifierOptions {
    hiddenSize: number;
    learningRate: number;
    momentum: number;
    maxGrad: number;
    confidenceThreshold: number;
    activation: 'relu' | 'swish';
}
export interface CommitClassifierTrainOptions {
    data: {
        text: string;
        category: Category;
    }[];
    epochs?: number;
}
export interface CommitClassifierContinueTrainingOptions extends CommitClassifierTrainOptions {
    filePath: string;
    decayFactor: number;
}
export declare class CommitClassifier {
    private brain;
    private vocabulary;
    private hiddenSize;
    private learningRate;
    private confidenceThreshold;
    private activation;
    private idfWeights;
    private datasetSize;
    constructor(options?: CommitClassifierOptions);
    private tokenize;
    private tokenizeToNgrams;
    train({ data, epochs, }: CommitClassifierTrainOptions): Promise<void>;
    continueTraining({ data, filePath, epochs, decayFactor, }: CommitClassifierContinueTrainingOptions): Promise<void>;
    getCommitQualityScore(text: string): number;
    classify(text: string): Category;
    private buildVocabulary;
    private textToVector;
    private calculateCosineSimilarity;
    generateChangelog(commits: string[]): string;
    private drawNeuralDashboard;
    saveModel(filePath: string): void;
    loadModel(filePath: string): boolean;
}
export {};
